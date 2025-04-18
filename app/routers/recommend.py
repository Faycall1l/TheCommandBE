from flask import Blueprint, request, jsonify
from app.utils.supabase_client import supabase

recommend_blueprint = Blueprint("recommend", __name__)

@recommend_blueprint.route("/recommendations", methods=["GET"])
def get_recommendations():
    niche_name = request.args.get("niche")
    
    if not niche_name:
        return jsonify({"error": "Missing 'niche' parameter"}), 400

    # Fetch the niche
    niche_data = supabase.table("niche").select("*").eq("niche_name", niche_name).execute()

    if not niche_data.data:
        return jsonify({"error": "Niche not found"}), 404

    niche_id = niche_data.data[0]["id"]

    # Fetch related data
    media_plan = supabase.table("media_plan").select("*").eq("niche_id", niche_id).execute()
    ideas = supabase.table("idea").select("*").eq("niche_id", niche_id).execute()
    peaks = supabase.table("peak").select("*").eq("niche_id", niche_id).execute()
    products = supabase.table("product").select("*").eq("niche_id", niche_id).order("number_of_sales", desc=True).limit(5).execute()

    return jsonify({
        "niche": niche_data.data[0],
        "top_products": products.data,
        "media_plan": media_plan.data,
        "ideas": ideas.data,
        "peaks": peaks.data
    })
