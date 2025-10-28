import zipfile
import requests
from pathlib import Path


DATA_SOURCES = {
    'pdf': {
        'url': 'https://drive.google.com/open?id=1GaiGj9SXXZ06NtxMl2jrYzUkO_ULTbDS&usp=drive_fs',
        'filename': 'pdf.zip',
        'extract_to': 'data/pdf'
    },
    'markdown': {
        'url': 'https://drive.google.com/open?id=1kiFkUfQeVKaAb70amawj4NPV-hUzRqCd&usp=drive_fs',
        'filename': 'markdown.zip',
        'extract_to': 'data/converted_md'
    },
    'paper_tags': {
        'url': 'https://drive.google.com/open?id=1gvJx4QhplaZRyWAXnW9bYgJSUGYF63_D&usp=drive_fs',
        'filename': 'paper_tags.json',
        'extract_to': 'data/paper_tags.json'
    },
    'qas': {
        'url': 'https://drive.google.com/open?id=1erEyVk3TTrWVLxkV7bRMXBjE7nDZ7INP&usp=drive_fs',
        'filename': 'qas.zip',
        'extract_to': 'data/qas'
    }
}


def extract_file_id(url):
    """Extract Google Drive file ID from URL"""
    if '/file/d/' in url:
        return url.split('/file/d/')[1].split('/')[0]
    elif 'id=' in url:
        return url.split('id=')[1].split('&')[0]
    return url


def download_from_google_drive(file_id, output_path):
    """Download file from Google Drive"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
            response = session.get(url, stream=True)
            break
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def main():
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    for name, config in DATA_SOURCES.items():
        print(f"\nDownloading {name}...")
        file_id = extract_file_id(config['url'])
        download_path = data_dir / config['filename']
        download_from_google_drive(file_id, download_path)
        
        if config['filename'].endswith('.zip'):
            print(f"Extracting to {config['extract_to']}...")
            Path(config['extract_to']).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(config['extract_to'])
            download_path.unlink()
        else:
            print(f"Saved to {download_path}")
    
    print("\nSetup complete!")


if __name__ == "__main__":
    main()
