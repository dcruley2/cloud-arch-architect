import os
import sys
import argparse
from google.cloud import aiplatform
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
from vertexai.preview import reasoning_engines

class TerraformAgent:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name
        self.chat_session = None

    def set_up(self):
        # Imports inside set_up ensure they are available in the remote environment
        from vertexai.generative_models import GenerativeModel
        
        system_instruction = (
            "You are an expert Google Cloud Solutions Architect specializing in Terraform. "
            "Your goal is to help users create infrastructure by:\n"
            "1. Asking clarifying questions if the deployment request is broad or lacks detail (e.g., asking about network visibility, machine types, or specific services).\n"
            "2. Generating high-quality, production-ready Terraform code.\n"
            "3. MANDATORY: You must use the variable placeholders \"${var.project_id}\" and \"${var.region}\" in your Terraform resources instead of hardcoded strings.\n"
            "4. Providing a variables.tf file structure that defines these placeholders."
        )
        
        self.model = GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
        )
        self.chat_session = self.model.start_chat()

    def query(self, message: str) -> str:
        if not self.chat_session:
            self.set_up()
        response = self.chat_session.send_message(message)
        return response.text

def create_bucket_if_not_exists(bucket_name, project, region):
    client = storage.Client(project=project)
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} already exists.")
    except:
        print(f"Creating bucket {bucket_name}...")
        bucket = client.create_bucket(bucket_name, location=region)
        print(f"Bucket {bucket_name} created.")
    return f"gs://{bucket_name}"

def main():
    parser = argparse.ArgumentParser(description="Cloud Arch Architect Agent CLI")
    parser.add_argument("--deploy", action="store_true", help="Deploy the agent to Vertex AI")
    parser.add_argument("--local", action="store_true", help="Run the agent locally for testing")
    args = parser.parse_args()

    project_id = "dcruley-demo-project"
    region = "us-central1"
    staging_bucket_name = f"{project_id}-agent-staging"

    aiplatform.init(project=project_id, location=region)

    if args.deploy:
        staging_bucket_uri = create_bucket_if_not_exists(staging_bucket_name, project_id, region)
        aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket_uri)
        
        print("Deploying TerraformAgent to Vertex AI Reasoning Engine...")
        agent = TerraformAgent()
        
        remote_agent = reasoning_engines.ReasoningEngine.create(
            agent,
            requirements=[
                "google-cloud-aiplatform[genai]",
                "cloudpickle",
                "vertexai"
            ],
            display_name="Cloud Arch Architect Agent",
        )
        print(f"Agent deployed successfully! Resource Name: {remote_agent.resource_name}")

    elif args.local:
        print("Starting local chat with TerraformAgent (Type 'exit' to quit)...")
        agent = TerraformAgent()
        agent.set_up()
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            response = agent.query(user_input)
            print(f"Agent: {response}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
