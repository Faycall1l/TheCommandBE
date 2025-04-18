from flask import Flask
from app.routers.forecast import forecast_blueprint
from app.routers.media import media_blueprint 
from app.routers.chat import chat_blueprint
from app.routers.recommend import recommend_blueprint
from app.utils.supabase_client import supabase



app = Flask(__name__)
app.register_blueprint(forecast_blueprint, url_prefix="/api/forecast")
app.register_blueprint(media_blueprint, url_prefix="/api/media")
app.register_blueprint(chat_blueprint, url_prefix="/api/chat")
app.register_blueprint(recommend_blueprint, url_prefix="/api")

if __name__ == '__main__':
    app.run(debug=True)
