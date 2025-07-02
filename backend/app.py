from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agent')))
from agent import run_team, arxiv_search
import asyncio

app = Flask(__name__)

@app.route('/api/literature_review', methods=['POST'])
def literature_review():
    data = request.json
    topic = data.get('topic', '')
    num_papers = int(data.get('num_papers', 2))
    # Use a similar logic as run_team, but for any topic and num_papers
    async def get_summary():
        from agent import teams
        task = f'Conduct a literature review on the topic - {topic} and return exactly {num_papers} papers.'
        last_content = None
        async for msg in teams.run_stream(task=task):
            if hasattr(msg, 'source') and getattr(msg, 'source', None) == 'SummarizerAgent' and hasattr(msg, 'content'):
                last_content = msg.content
        return last_content
    try:
        summary = asyncio.run(get_summary())
        return jsonify({'content': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
