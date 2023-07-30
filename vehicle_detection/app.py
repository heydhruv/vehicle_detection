from base import app

# app = Flask(__name__)


if __name__ == '__main__':
    app.run(debug=True, port=8081, threaded=True)
