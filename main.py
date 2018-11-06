import flaskapp as app
import train

train.train()
app.app.run(host='127.0.0.1', port=9055, debug=True, threaded=True)