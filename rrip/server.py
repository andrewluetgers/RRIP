"""
Tile Server - HTTP server for efficient tile serving
"""

from flask import Flask, send_file, jsonify, request
from io import BytesIO
from .tile_manager import TileManager


def create_app(storage_dir):
    """
    Create Flask app for tile serving.
    
    Args:
        storage_dir: Directory containing compressed tiles
        
    Returns:
        Flask app
    """
    app = Flask(__name__)
    tile_manager = TileManager(storage_dir)
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'ok'})
    
    @app.route('/images')
    def list_images():
        """List all available images."""
        images = tile_manager.list_images()
        return jsonify({'images': images})
    
    @app.route('/images/<image_id>')
    def get_image_info(image_id):
        """Get image metadata."""
        info = tile_manager.get_image_info(image_id)
        if info is None:
            return jsonify({'error': 'Image not found'}), 404
        return jsonify(info)
    
    @app.route('/images/<image_id>/tile/<int:tile_index>')
    def get_tile(image_id, tile_index):
        """
        Get a specific tile by index.
        
        Returns PNG image.
        """
        try:
            tile = tile_manager.get_tile(image_id, tile_index)
            
            # Convert to PNG and return
            img_io = BytesIO()
            tile.save(img_io, 'PNG')
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
    
    @app.route('/images/<image_id>/tile_at')
    def get_tile_at_position(image_id):
        """
        Get tile at specific position.
        
        Query params: x, y
        Returns PNG image.
        """
        try:
            x = int(request.args.get('x', 0))
            y = int(request.args.get('y', 0))
            
            tile = tile_manager.get_tile_by_position(image_id, x, y)
            
            # Convert to PNG and return
            img_io = BytesIO()
            tile.save(img_io, 'PNG')
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
    
    @app.route('/images/<image_id>/full')
    def get_full_image(image_id):
        """
        Get full reconstructed image.
        
        Returns PNG image.
        """
        try:
            image = tile_manager.get_full_image(image_id)
            
            # Convert to PNG and return
            img_io = BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
    
    return app


def run_server(storage_dir, host='0.0.0.0', port=5000, debug=False):
    """
    Run the tile server.
    
    Args:
        storage_dir: Directory containing compressed tiles
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    app = create_app(storage_dir)
    app.run(host=host, port=port, debug=debug)
