import os
import base64
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
    )

client = OpenAI(api_key=api_key)


@app.get("/")
def home():
    return jsonify({
        "ok": True,
        "service": "음식요정 AI 서버",
        "version": "3.0"
    })


@app.post("/analyze-food")
def analyze_food():

    try:

        body = request.get_json()

        if not body:
            return jsonify({
                "error": "JSON 데이터가 없습니다."
            }), 400

        image = body.get("image")

        if not image:
            return jsonify({
                "error": "이미지가 없습니다."
            }), 400

        # data:image/jpeg;base64,... 형태인지 확인
        if "," not in image:
            return jsonify({
                "error": "올바른 이미지 데이터가 아닙니다."
            }), 400

        prompt = """
사진 속에서 확인할 수 있는 음식 또는 식재료를 찾아주세요.

중요:
- 음식 이름만 찾아주세요.
- 사람, 접시, 식탁, 주방용품 등은 음식으로 인식하지 마세요.
- 확실하지 않은 것은 억지로 추측하지 마세요.
- 한국어로 답하세요.
- 가능한 경우 계란, 우유, 참기름, 간장처럼
  실제 식재료의 구체적인 이름을 사용하세요.

반드시 아래 JSON 형식으로만 답하세요.

{
  "foods": [
    {
      "name": "계란",
      "confidence": 0.95
    }
  ]
}

음식을 확실하게 찾을 수 없다면:

{
  "foods": []
}
"""

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": image
                        }
                    ]
                }
            ]
        )

        text = response.output_text.strip()

        # ```json ... ``` 제거
        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        result = json.loads(text)

        foods = result.get("foods", [])

        cleaned = []

        for food in foods:

            name = str(
                food.get("name", "")
            ).strip()

            confidence = float(
                food.get("confidence", 0)
            )

            if name and confidence >= 0.55:

                cleaned.append({
                    "name": name,
                    "confidence": confidence
                })

        return jsonify({
            "foods": cleaned
        })

    except Exception as e:

        print("AI ERROR:", repr(e))

        return jsonify({
            "error": "음식 인식에 실패했습니다.",
            "detail": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8787,
        debug=True
    )