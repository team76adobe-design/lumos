import os
import urllib.request
import urllib.error
from urllib.parse import urlparse
import time

# Define the images with their names and URLs
images = {
    "Nature Reclaims": "https://thumbs.dreamstime.com/b/lush-green-ferns-overlook-dense-urban-cityscape-night-glowing-lights-dark-moody-aesthetic-backgrounds-striking-377593262.jpg",
    "Masterpiece Oil": "https://cdn.magicdecor.in/com/2024/07/09170149/Woman-Oil-Painting-Wallpaper-Mural.jpg",
    "Japanese Comic Art": "https://www.shutterstock.com/image-vector/beautiful-anime-girl-landscape-wallpaper-600nw-2321703851.jpg",
    "Cosmic Nebula": "https://4kwallpapers.com/images/walls/thumbs_2t/2469.jpg",
    "Arcane Steel Magic": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScx_P7rFZaRHDHVMtqh-wkjm37Ns91Np2GLA&s",
    "16-Bit Arcade": "https://i.pinimg.com/736x/7a/67/d4/7a67d4c54601d570af4b20e952b0a8bc.jpg",
    "Studio Ghibli Film": "https://getwallpapers.com/wallpaper/full/9/2/4/654076.jpg",
    "Ethereal Portrait": "https://models-online-persist.shakker.cloud/img/78d39a8509f94f09947a1f6e3758d2db/16483db9f61e32e6f92af1afae749b3d94475aa86a57ce4114008f3ee9af82d3.jpeg?x-oss-process=image/resize,w_764,m_lfit/format,webp",
    "Grand Theft Auto": "https://preview.redd.it/i-made-an-a3-poster-of-gta-6-v0-oi5oirpvht6c1.png?auto=webp&s=6578423d748421e04e50395ea0a502ff71f11717",
    "Post-Apocalyptic Zombie": "https://www.zbrushcentral.com/uploads/default/original/4X/6/3/4/63448e5c7a310ff4dc3b22f24f4f7f61508a3bf2.jpeg",
    "Pixar 3D Toy": "https://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg",
    "Cyberpunk Neon": "https://wallpapercave.com/wp/wp13293194.jpg",
    "Pencil Sketch": "https://www.morphico.com/images/13327047-thumb.jpg",
    "Impressionist Brush": "https://img.freepik.com/premium-photo/paint-brushes-palette-with-colorful-paints-grunge-background-colorful-artist-brushes-paint-ai-generated_538213-5536.jpg",
    "Default Placeholder": "https://media.niftygateway.com/video/upload/q_auto:good,w_500/v1618083696/A/Vexx/DOODE_1_xc9hb8.jpeg"
}

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_file_extension(url):
    """Extract file extension from URL"""
    parsed_url = urlparse(url.split('?')[0])  # Remove query parameters
    path = parsed_url.path
    if path:
        return os.path.splitext(path)[1] or ".jpg"
    return ".jpg"

def download_image(name, url):
    """Download an image from URL and save it"""
    try:
        print(f"Downloading: {name}...", end=" ")
        
        # Set headers to avoid 403 Forbidden
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=10)
        
        # Get file extension
        ext = get_file_extension(url)
        
        # Create safe filename
        safe_name = name.replace(" ", "_").replace("/", "_").lower()
        filename = f"{safe_name}{ext}"
        filepath = os.path.join(script_dir, filename)
        
        # Save the image
        with open(filepath, 'wb') as f:
            f.write(response.read())
        
        file_size = os.path.getsize(filepath) / 1024  # Size in KB
        print(f"✓ Saved as {filename} ({file_size:.1f} KB)")
        return filepath
        
    except urllib.error.URLError as e:
        print(f"✗ Failed: Connection error - {str(e)}")
        return None
    except urllib.error.HTTPError as e:
        print(f"✗ Failed: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None

def main():
    print("=" * 60)
    print("Image Downloader")
    print("=" * 60)
    print(f"Target directory: {script_dir}\n")
    
    successful = 0
    failed = 0
    
    for name, url in images.items():
        result = download_image(name, url)
        if result:
            successful += 1
        else:
            failed += 1
        time.sleep(1)  # Rate limiting between downloads
    
    print("\n" + "=" * 60)
    print(f"Download Complete!")
    print(f"Successful: {successful}/{len(images)}")
    print(f"Failed: {failed}/{len(images)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
