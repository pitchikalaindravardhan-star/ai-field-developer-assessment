"""Same server, but the tool handlers misbehave the way real code does."""

import logging
import os
import subprocess
import sys

from task1_mcp_server import server


async def noisy_call_tool(ctx, params):
    print("debug: print() from a handler")
    sys.stdout.write("debug: sys.stdout.write\n")
    sys.stdout.flush()
    os.write(1, b"debug: raw write to fd 1\n")
    subprocess.run(["echo", "debug: child process output"], check=True)
    logging.getLogger("noisy").info("this one belongs on stderr")
    return await original(ctx, params)


original = server.call_tool
server.server = server.Server(
    "customer-support",
    version="1.0.0",
    on_list_tools=server.list_tools,
    on_call_tool=noisy_call_tool,
)

if __name__ == "__main__":
    server.main()
