from flask import Blueprint, request, jsonify
from app.services.media_planner import generate_media_plan
import json
from app.utils.supabase_client import supabase

media_blueprint = Blueprint("media", __name__)

@media_blueprint.route("/plan", methods=["POST"])
def get_media_plan():
    try:
        data = request.get_json()
        forecast_data = data.get("forecast")
        niche = data.get("niche", "general")
        niche_id = data.get("niche_id", None)

        if not forecast_data:
            return jsonify({"error": "Missing forecast data"}), 400

        if not niche_id:
            niche_object = supabase.table("niche").select("*").eq("niche_name", niche).execute()
            if not niche_object.data:
                return jsonify({"error": "Niche not found"}), 400
            else:
                niche_id = niche_object.data[0]["id"]

        # Optional: convert forecast JSON into readable text
        text = "\n".join([
            f"{row['ds']} → {int(row['yhat'])} units"
            for row in forecast_data
        ])

        plan = generate_media_plan(forecast_data=text, niche=niche)

        if niche_id:
            # Save the media plan to the database
            supabase.table("media_plan").insert({
                "title": f"Media Plan for {niche}",
                "description": json.dumps(plan),
                "niche_id": niche_id,
            }).execute()


        return jsonify({"media_plan": plan})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
