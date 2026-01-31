"""
Test the HTTP tile server API
"""

import time
import requests
from PIL import Image
import subprocess
import signal
import os
from rrip import TileManager

print("Setting up test data...")

# Create tile manager and store a test image
storage_dir = '/tmp/test_server_storage'
manager = TileManager(storage_dir)

# Create a simple test image
test_image = Image.new('RGB', (512, 512))
pixels = test_image.load()
for i in range(512):
    for j in range(512):
        pixels[j, i] = (i % 256, j % 256, 128)

# Store image
config = {'downsample_factor': 4, 'quality': 60, 'tile_size': 256}
manager.store_image('test_api_image', test_image, encoder_config=config)

print("Test data prepared")

# Start server in background
print("\nStarting server...")
server_process = subprocess.Popen(
    ['rrip', 'serve', storage_dir, '--port', '5556'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start with retry loop
print("Waiting for server to be ready...")
base_url = 'http://localhost:5556'
max_retries = 15
retry_delay = 0.5

server_ready = False
for attempt in range(max_retries):
    try:
        response = requests.get(f'{base_url}/health', timeout=1)
        if response.status_code == 200:
            server_ready = True
            print(f"Server ready after {attempt + 1} attempts")
            break
    except (requests.ConnectionError, requests.Timeout):
        pass
    time.sleep(retry_delay)

if not server_ready:
    print("ERROR: Server failed to start within timeout")
    server_process.terminate()
    server_process.wait(timeout=5)
    exit(1)

try:
    base_url = 'http://localhost:5556'
    
    # Test health endpoint
    print("\nTesting /health endpoint...")
    response = requests.get(f'{base_url}/health', timeout=2)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    
    # Test list images endpoint
    print("\nTesting /images endpoint...")
    response = requests.get(f'{base_url}/images', timeout=2)
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Images: {data['images']}")
    
    # Test get image info endpoint
    print("\nTesting /images/<id> endpoint...")
    response = requests.get(f'{base_url}/images/test_api_image', timeout=2)
    print(f"  Status: {response.status_code}")
    info = response.json()
    print(f"  Image size: {info['image_size']}")
    print(f"  Number of tiles: {info['num_tiles']}")
    
    # Test get tile endpoint
    print("\nTesting /images/<id>/tile/<index> endpoint...")
    response = requests.get(f'{base_url}/images/test_api_image/tile/0', timeout=2)
    print(f"  Status: {response.status_code}")
    print(f"  Content-Type: {response.headers.get('Content-Type')}")
    print(f"  Content size: {len(response.content)} bytes")
    
    # Save tile
    with open('/tmp/api_tile_0.png', 'wb') as f:
        f.write(response.content)
    print(f"  Tile saved to /tmp/api_tile_0.png")
    
    # Test get tile at position endpoint
    print("\nTesting /images/<id>/tile_at?x=&y= endpoint...")
    response = requests.get(f'{base_url}/images/test_api_image/tile_at?x=300&y=300', timeout=2)
    print(f"  Status: {response.status_code}")
    print(f"  Content-Type: {response.headers.get('Content-Type')}")
    print(f"  Content size: {len(response.content)} bytes")
    
    # Test get full image endpoint
    print("\nTesting /images/<id>/full endpoint...")
    response = requests.get(f'{base_url}/images/test_api_image/full', timeout=2)
    print(f"  Status: {response.status_code}")
    print(f"  Content-Type: {response.headers.get('Content-Type')}")
    print(f"  Content size: {len(response.content)} bytes")
    
    # Save full image
    with open('/tmp/api_full_image.png', 'wb') as f:
        f.write(response.content)
    print(f"  Full image saved to /tmp/api_full_image.png")
    
    print("\n" + "="*50)
    print("All API tests passed successfully!")
    print("="*50)
    
except Exception as e:
    print(f"\nError during testing: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Stop server
    print("\nStopping server...")
    server_process.terminate()
    server_process.wait(timeout=5)
    print("Server stopped")
