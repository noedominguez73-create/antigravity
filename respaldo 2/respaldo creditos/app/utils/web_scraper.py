import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
from datetime import datetime

logger = logging.getLogger(__name__)

class WebScraper:
    """
    Utility class for scraping and cleaning web content from allowed domains.
    """
    
    MAX_CHARS = 50000
    TIMEOUT = 10  # seconds
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    ALLOWED_DOMAINS = {
        'imss.gob.mx',
        'www.imss.gob.mx',
        'sat.gob.mx',
        'www.sat.gob.mx',
        'gob.mx',
        'www.gob.mx',
        'dof.gob.mx',
        'www.dof.gob.mx',
        'diputados.gob.mx',
        'www.diputados.gob.mx',
        'senado.gob.mx',
        'www.senado.gob.mx'
    }

    @staticmethod
    def is_allowed_domain(url):
        """Check if the URL belongs to an allowed domain."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            return domain in WebScraper.ALLOWED_DOMAINS
        except Exception:
            return False

    @staticmethod
    def fetch_and_clean(url):
        """
        Fetches the URL and extracts the main text content.
        Returns a dictionary with success status and content/error.
        """
        try:
            # 1. Validate domain
            if not WebScraper.is_allowed_domain(url):
                return {
                    "success": False,
                    "error": "Dominio no permitido. Solo sitios oficiales (.gob.mx)",
                    "fetched_at": datetime.utcnow().isoformat()
                }

            # 2. Fetch content
            response = requests.get(url, headers=WebScraper.HEADERS, timeout=WebScraper.TIMEOUT)
            response.raise_for_status()
            
            # 3. Parse and clean
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script, style, and navigation elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                script.decompose()
            
            # Get title
            title = soup.title.string.strip() if soup.title else url
            
            # Get text with better encoding handling
            if response.encoding is None:
                response.encoding = response.apparent_encoding
                
            text = soup.get_text(separator=' ', strip=True)
            
            # Fallback: If text is too short, try meta description
            if len(text) < 200:
                meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                if meta_desc and meta_desc.get('content'):
                    text += f"\n\nDescripción: {meta_desc['content']}"
            
            # 4. Limit content
            if len(text) > WebScraper.MAX_CHARS:
                text = text[:WebScraper.MAX_CHARS] + "... (truncated)"
                
            return {
                "success": True,
                "content": f"Title: {title}\n\n{text}",
                "fetched_at": datetime.utcnow().isoformat()
            }
            
        except requests.Timeout:
            return {
                "success": False,
                "error": "Tiempo de espera agotado al conectar con el sitio",
                "fetched_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return {
                "success": False,
                "error": f"Error obteniendo contenido: {str(e)}",
                "fetched_at": datetime.utcnow().isoformat()
            }
