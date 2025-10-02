from llama_cpp.server.app import create_app
from llama_cpp import Llama
import uvicorn

app = create_app(
    llama=Llama(
        model_path="/workspaces/POC_AI/models/models/deepseek-llm-7b-chat.Q4_K_M.gguf",
        chat_format="chatml"
    )
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
