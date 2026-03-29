# server.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import threading
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai

from functools import wraps
from config import Config
from gemini_client import GeminiClient
from prompts import Prompts
from data_preprocessor import DataPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from predictor import MoodPredictor
from auth import Auth, FriendManager

app = Flask(__name__)
CORS(app, origins=Config.CLIENT_ORIGIN)

# ── shared clients ─────────────────────────────────────────────
gemini_connection = genai.Client(api_key=Config.GEMINI_API_KEY)
client = GeminiClient(gemini_connection, Config.GEMINI_MODEL)
sa     = SentimentAnalyzer()

# ── in-memory storage (per-user) ───────────────────────────────
user_entries = {}  # username -> [entries]

# ── prediction cache (per-user) ────────────────────────────────
prediction_cache  = {}  # username -> result
prediction_status = {}  # username -> {"status": ...}

# ── friend alerts (per-user inbox) ─────────────────────────────
friend_alerts = {}  # username -> [alert_dict, ...]


def _get_entries(username: str) -> list:
    return user_entries.setdefault(username, [])


def _check_consecutive_low_mood(username: str, threshold: int = 2, days: int = 5) -> bool:
    """Check if user has `days` consecutive entries with mood_score <= threshold."""
    entries = _get_entries(username)
    print(f"[ALERT CHECK] {username}: {len(entries)} entries total")
    if len(entries) < days:
        print(f"[ALERT CHECK] {username}: Not enough entries ({len(entries)} < {days})")
        return False
    # sort by date, take last `days`
    sorted_entries = sorted(entries, key=lambda e: e['date'])
    last_n = sorted_entries[-days:]
    scores = [e['mood_score'] for e in last_n]
    dates  = [e['date'] for e in last_n]
    print(f"[ALERT CHECK] {username}: Last {days} scores={scores}, dates={dates}")
    # all must be <= threshold
    if not all(e['mood_score'] <= threshold for e in last_n):
        print(f"[ALERT CHECK] {username}: Not all scores <= {threshold}")
        return False
    # must be truly consecutive dates
    from datetime import datetime, timedelta
    for i in range(1, len(last_n)):
        d1 = datetime.strptime(last_n[i-1]['date'], '%Y-%m-%d')
        d2 = datetime.strptime(last_n[i]['date'], '%Y-%m-%d')
        if (d2 - d1).days != 1:
            print(f"[ALERT CHECK] {username}: Dates not consecutive ({last_n[i-1]['date']} -> {last_n[i]['date']})")
            return False
    print(f"[ALERT CHECK] {username}: ✅ TRIGGERED! 5 consecutive low mood days")
    return True


def _send_friend_alerts(username: str):
    """If user has 5 consecutive low-mood days, alert all their friends."""
    if not _check_consecutive_low_mood(username):
        return False
    friends_data = FriendManager.get_friends(username)
    friends = friends_data.get('friends', [])
    if not friends:
        print(f"[ALERT] {username} has no friends, skipping alert")
        return False

    from datetime import datetime
    sorted_entries = sorted(_get_entries(username), key=lambda e: e['date'])
    last_5 = sorted_entries[-5:]
    avg_score = round(sum(e['mood_score'] for e in last_5) / 5, 1)

    alert = {
        'id':        f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'from_user': username,
        'type':      'low_mood_warning',
        'message':   f'{username} has been feeling down for a few days (avg: {avg_score}/5). A small message from you might brighten their day.',
        'avg_score': avg_score,
        'date_range': f"{last_5[0]['date']} ~ {last_5[-1]['date']}",
        'created_at': datetime.now().isoformat(),
        'dismissed':  False,
    }

    sent_to = []
    for friend in friends:
        inbox = friend_alerts.setdefault(friend, [])
        # don't duplicate: skip if we already alerted about this user today
        already = any(
            a['from_user'] == username and a['date_range'] == alert['date_range']
            for a in inbox
        )
        if not already:
            inbox.append(alert.copy())
            sent_to.append(friend)
            print(f"[ALERT] ✅ Sent alert to {friend}: {username} low mood")
        else:
            print(f"[ALERT] Skipped {friend}: duplicate alert already exists")

    return len(sent_to) > 0


# ── auth decorator ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        username = Auth.verify_token(token)
        if not username:
            return jsonify({"error": "未登录或 token 无效"}), 401
        request.username = username
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════
# AUTH — login / register / logout
# ══════════════════════════════════════════════════════════════
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    result = Auth.login(
        data.get('username', ''),
        data.get('password', ''),
    )
    status = 200 if result['success'] else 401
    return jsonify(result), status


@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    result = Auth.register(
        data.get('username', ''),
        data.get('password', ''),
        data.get('nickname', ''),
    )
    status = 200 if result['success'] else 400
    return jsonify(result), status


@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    Auth.logout(token)
    return jsonify({"success": True, "message": "已退出登录"})


@app.route('/api/auth/me', methods=['GET'])
@login_required
def me():
    return jsonify({"username": request.username})


# ══════════════════════════════════════════════════════════════
# FRIENDS — send / accept / reject / remove / list
# ══════════════════════════════════════════════════════════════
@app.route('/api/friends', methods=['GET'])
@login_required
def list_friends():
    return jsonify(FriendManager.get_friends(request.username))


@app.route('/api/friends/request', methods=['POST'])
@login_required
def send_friend_request():
    to_user = request.json.get('to_user', '')
    return jsonify(FriendManager.send_request(request.username, to_user))


@app.route('/api/friends/accept', methods=['POST'])
@login_required
def accept_friend():
    from_user = request.json.get('from_user', '')
    return jsonify(FriendManager.accept_request(request.username, from_user))


@app.route('/api/friends/reject', methods=['POST'])
@login_required
def reject_friend():
    from_user = request.json.get('from_user', '')
    return jsonify(FriendManager.reject_request(request.username, from_user))


@app.route('/api/friends/remove', methods=['POST'])
@login_required
def remove_friend():
    friend = request.json.get('friend', '')
    return jsonify(FriendManager.remove_friend(request.username, friend))


# ══════════════════════════════════════════════════════════════
# FRIEND ALERTS — low mood warnings
# ══════════════════════════════════════════════════════════════
@app.route('/api/friend-alerts', methods=['GET'])
@login_required
def get_friend_alerts():
    """Return active (non-dismissed) alerts for the current user."""
    inbox = friend_alerts.get(request.username, [])
    active = [a for a in inbox if not a.get('dismissed')]
    return jsonify(active)


@app.route('/api/friend-alerts/dismiss', methods=['POST'])
@login_required
def dismiss_friend_alert():
    """Dismiss a specific alert by id."""
    alert_id = request.json.get('alert_id', '')
    inbox = friend_alerts.get(request.username, [])
    for a in inbox:
        if a['id'] == alert_id:
            a['dismissed'] = True
            break
    return jsonify({"success": True})


@app.route('/api/friend-alerts/dismiss-all', methods=['POST'])
@login_required
def dismiss_all_friend_alerts():
    """Dismiss all alerts."""
    inbox = friend_alerts.get(request.username, [])
    for a in inbox:
        a['dismissed'] = True
    return jsonify({"success": True})


# ══════════════════════════════════════════════════════════════
# DEBUG — check system state (remove in production)
# ══════════════════════════════════════════════════════════════
@app.route('/api/debug/state', methods=['GET'])
def debug_state():
    return jsonify({
        'user_entries': {k: len(v) for k, v in user_entries.items()},
        'friend_alerts': {k: len(v) for k, v in friend_alerts.items()},
        'alert_details': {k: [{'from': a['from_user'], 'dismissed': a['dismissed'], 'date_range': a['date_range']} for a in v] for k, v in friend_alerts.items()},
    })


# ══════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/friend')
def friend_page():
    return render_template('friend.html')


# ══════════════════════════════════════════════════════════════
# API — friend mood summary (privacy-safe, no journal text)
# ══════════════════════════════════════════════════════════════
@app.route('/api/friends/mood-summary', methods=['GET'])
@login_required
def friend_mood_summary():
    friends_data = FriendManager.get_friends(request.username)
    friends = friends_data.get('friends', [])
    summaries = []
    for friend in friends:
        fentries = _get_entries(friend)
        if not fentries:
            summaries.append({'username': friend, 'total_entries': 0,
                              'recent_scores': [], 'avg_mood': None, 'streak': 0,
                              'last_entry_date': None, 'mood_trend': 'none'})
            continue
        sorted_e = sorted(fentries, key=lambda e: e['date'])
        recent = sorted_e[-7:]
        scores = [e['mood_score'] for e in recent]
        all_scores = [e['mood_score'] for e in sorted_e]
        avg = round(sum(all_scores) / len(all_scores), 1)
        from datetime import datetime, timedelta
        streak = 0
        check = datetime.now()
        dates_set = {e['date'] for e in sorted_e}
        while check.strftime('%Y-%m-%d') in dates_set:
            streak += 1
            check -= timedelta(days=1)
        trend = 'stable'
        if len(scores) >= 6:
            prev3 = sum(scores[-6:-3]) / 3
            last3 = sum(scores[-3:]) / 3
            if last3 - prev3 > 0.5: trend = 'improving'
            elif prev3 - last3 > 0.5: trend = 'declining'
        consecutive_low = 0
        for e in reversed(sorted_e):
            if e['mood_score'] <= 2: consecutive_low += 1
            else: break
        summaries.append({
            'username': friend, 'total_entries': len(sorted_e),
            'recent_scores': [{'date': e['date'], 'score': e['mood_score']} for e in recent],
            'avg_mood': avg, 'streak': streak,
            'last_entry_date': sorted_e[-1]['date'], 'mood_trend': trend,
            'consecutive_low_days': consecutive_low,
            'needs_support': consecutive_low >= 5,
        })
    return jsonify(summaries)


# ══════════════════════════════════════════════════════════════
# API — submit a journal entry
# ══════════════════════════════════════════════════════════════
@app.route('/api/submit', methods=['POST'])
@login_required
def submit():
    data        = request.json
    journal     = data.get('journal_text', '')
    mood_score  = int(data.get('mood_score', 3))
    date        = data.get('date', '')
    time_of_day = data.get('time_of_day', 'evening')
    username    = request.username
    entries     = _get_entries(username)

    # sentiment analysis (local, instant)
    result = sa.analyze(journal)

    # reflection prompt from Gemini
    from datetime import datetime
    try:
        day_of_week = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
    except Exception:
        day_of_week = 'Monday'

    reflection = client.ask(
        Prompts.REFLECTION,
        f"Mood score: {mood_score}/5\n"
        f"Day of week: {day_of_week}\n"
        f"Journal entry: {journal}"
    )

    # store entry
    entry = {
        'date':            date,
        'mood_score':      mood_score,
        'journal_text':    journal,
        'time_of_day':     time_of_day,
        'sentiment_label': result.label,
        'sentiment_note':  result.brief_note,
        'sentiment_flags': result.flags,
        'sentiment_score': result.score,
    }
    entries.append(entry)

    # ── friend low-mood alert check ────────────────────────────
    alert_triggered = _send_friend_alerts(username)

    # reset prediction cache when new data comes in
    prediction_status.setdefault(username, {})["status"] = "idle"
    prediction_cache.pop(username, None)

    # escalation check
    # only use CURRENT entry flags — prevents old flags from re-triggering on new positive entries
    recent_scores = [e['mood_score'] for e in entries[-5:]]
    current_flags = result.flags
    needs_esc     = sa.should_escalate(recent_scores, current_flags)

    escalation = None
    if needs_esc:
        escalation = client.ask_json(
            Prompts.ESCALATION,
            json.dumps({"last_5_scores": recent_scores, "sentiment_flags": current_flags})
        )

    return jsonify({
        'sentiment': {
            'label':      result.label,
            'score':      result.score,
            'brief_note': result.brief_note,
            'flags':      result.flags,
        },
        'reflection':    reflection,
        'escalation':    escalation,
        'total_entries': len(entries),
        'alert_triggered': alert_triggered,
    })


# ══════════════════════════════════════════════════════════════
# API — get all entries
# ══════════════════════════════════════════════════════════════
@app.route('/api/entries', methods=['GET'])
@login_required
def get_entries():
    return jsonify(_get_entries(request.username))


# ══════════════════════════════════════════════════════════════
# API — pattern analysis (needs 7+ entries)
# ══════════════════════════════════════════════════════════════
@app.route('/api/patterns', methods=['GET'])
@login_required
def patterns():
    entries = _get_entries(request.username)
    if len(entries) < 7:
        return jsonify({
            'error': f'Need at least 7 entries. You have {len(entries)} so far.'
        }), 400

    last_14   = entries[-14:]
    mood_data = [
        {
            'score':      e['mood_score'],
            'day':        pd.to_datetime(e['date']).strftime('%a'),
            'brief_note': e['sentiment_note'],
        }
        for e in last_14
    ]
    analysis = client.ask_json(Prompts.PATTERN_ANALYSIS, json.dumps(mood_data))
    return jsonify(analysis)


# ══════════════════════════════════════════════════════════════
# API — LSTM prediction (runs in background thread)
# ══════════════════════════════════════════════════════════════
@app.route('/api/predict', methods=['GET'])
@login_required
def predict():
    username = request.username
    entries  = _get_entries(username)
    if len(entries) < 4:
        return jsonify({
            'error': f'Need at least 4 entries. You have {len(entries)} so far.'
        }), 400

    user_status = prediction_status.get(username, {})

    # already have a cached result
    if user_status.get("status") == "done":
        return jsonify(prediction_cache[username])

    # already training
    if user_status.get("status") == "training":
        return jsonify({
            'status':  'training',
            'message': 'Still training, check back soon...'
        })

    # start training in background thread
    prediction_status.setdefault(username, {})["status"] = "training"

    def train_and_predict():
        try:
            scores = [e['mood_score'] for e in entries]
            dates  = [e['date'] for e in entries]

            predictor = MoodPredictor(
                window_size=3,
                hidden_size=16,
                epochs=50,
                lr=0.01
            )
            predictor.train(scores)
            future_scores = predictor.predict(scores, days=7)

            from datetime import datetime, timedelta
            last_date    = datetime.strptime(dates[-1], '%Y-%m-%d')
            future_dates = [
                (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(7)
            ]

            prediction_cache[username] = {
                'status':            'done',
                'historical_dates':  dates,
                'historical_scores': scores,
                'predicted_dates':   future_dates,
                'predicted_scores':  future_scores,
            }
            prediction_status[username]["status"] = "done"
            print(f"LSTM training complete for {username}.")

        except Exception as e:
            prediction_status[username]["status"] = "error"
            prediction_cache[username] = {'status': 'error', 'error': str(e)}
            print(f"LSTM error for {username}: {e}")

    thread        = threading.Thread(target=train_and_predict)
    thread.daemon = True
    thread.start()

    return jsonify({
        'status':  'training',
        'message': 'Training started! Check back in about 10 seconds.'
    })


# ══════════════════════════════════════════════════════════════
# API — poll prediction status
# ══════════════════════════════════════════════════════════════
@app.route('/api/predict/status', methods=['GET'])
@login_required
def predict_status_route():
    username    = request.username
    user_status = prediction_status.get(username, {})
    if user_status.get("status") == "done":
        return jsonify(prediction_cache.get(username, {}))
    return jsonify({'status': user_status.get("status", "idle")})


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# CHAT SYSTEM
# ══════════════════════════════════════════════════════════════
# chat_messages: { "user1_user2": [ {from, text, timestamp} ] }
chat_messages = {}

def chat_key(u1, u2):
    return '_'.join(sorted([u1, u2]))

@app.route('/api/chat/<friend>', methods=['GET'])
@login_required
def get_chat(friend):
    username = request.username
    key = chat_key(username, friend)
    msgs = chat_messages.get(key, [])
    # only return messages after 'since' timestamp if provided
    since = request.args.get('since', '')
    if since:
        msgs = [m for m in msgs if m['timestamp'] > since]
    return jsonify(msgs)

@app.route('/api/chat/<friend>', methods=['POST'])
@login_required
def send_chat(friend):
    username = request.username
    text = request.json.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Empty message'}), 400
    from datetime import datetime
    msg = {
        'from': username,
        'text': text,
        'timestamp': datetime.now().isoformat()
    }
    key = chat_key(username, friend)
    chat_messages.setdefault(key, []).append(msg)
    return jsonify({'success': True, 'message': msg})


# ══════════════════════════════════════════════════════════════
# FRIEND ALERTS (for friend.html)
# ══════════════════════════════════════════════════════════════
@app.route('/api/friend-alerts', methods=['GET'])
@login_required
def friend_alerts():
    username = request.username
    friends_data = FriendManager.get_friends(username)
    friends = friends_data.get('friends', [])
    alerts = []
    for friend in friends:
        friend_entries = user_entries.get(friend, [])
        if len(friend_entries) < 5:
            continue
        last_5 = friend_entries[-5:]
        low = [e for e in last_5 if e['mood_score'] <= 2]
        if len(low) == 5:
            import json as _json
            try:
                with open('users.json', 'r') as f:
                    db = _json.load(f)
                friend_nickname = db['users'].get(friend, {}).get('nickname', friend)
            except:
                friend_nickname = friend
            alerts.append({
                'id': friend,
                'friend_username': friend,
                'friend_name': friend_nickname,
                'friend_avatar': '🌸',
                'days': [{'date': e['date'], 'score': e['mood_score']} for e in last_5]
            })
    return jsonify(alerts)

@app.route('/api/friend-alerts/dismiss', methods=['POST'])
@login_required
def dismiss_alert():
    return jsonify({'success': True})

@app.route('/api/friends/mood-summary', methods=['GET'])
@login_required  
def mood_summary():
    username = request.username
    friends_data = FriendManager.get_friends(username)
    friends = friends_data.get('friends', [])
    summary = []
    for friend in friends:
        entries = user_entries.get(friend, [])
        last = entries[-1] if entries else None
        summary.append({
            'username': friend,
            'last_score': last['mood_score'] if last else None,
            'last_date': last['date'] if last else None,
            'total_entries': len(entries)
        })
    return jsonify(summary)


# ══════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/friend')
def friend_page():
    return render_template('friend.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", Config.PORT))
    print(f"Server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
