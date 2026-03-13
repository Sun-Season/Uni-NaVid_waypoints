#!/usr/bin/env python3
"""Server startup script."""

import argparse
import uvicorn

from server.main import app, set_model_paths


def main():
    parser = argparse.ArgumentParser(description='Start the navigation API server')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/mnt/dataset/wj_zqc/VLN/model/uninavid-7b-full-224-video-fps-1-grid-2',
        help='Path to the base model'
    )
    parser.add_argument(
        '--lora_path',
        type=str,
        default='output/vln_action_text_test',
        help='Path to the LoRA weights (optional)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (use 0.0.0.0 for external access)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to listen on'
    )
    args = parser.parse_args()

    # Set model paths
    set_model_paths(args.model_path, args.lora_path)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"API docs available at: http://{args.host}:{args.port}/docs")

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == '__main__':
    main()
