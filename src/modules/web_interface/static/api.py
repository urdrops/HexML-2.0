from flask import Flask, send_file, jsonify
import threading

app = Flask(__name__)

animation_state = 'on'


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/animation_state')
def get_animation_state():
    global animation_state
    return jsonify({'state': animation_state})


@app.route('/toggle_animation/<state>')
def toggle_animation(state):
    global animation_state
    if state in ['on', 'off']:
        animation_state = state
        return jsonify({'success': True, 'state': animation_state})
    return jsonify({'success': False, 'error': 'Invalid state'})


def run_server():
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    while True:
        command = input("Введите 'on' для включения анимации или 'off' для выключения (q для выхода): ")
        if command.lower() == 'q':
            break
        elif command in ['on', 'off']:
            animation_state = command
            print(f"Анимация {'включена' if command == 'on' else 'выключена'}")
        else:
            print("Неверная команда. Используйте 'on' или 'off'.")

    print("Выход из программы...")
