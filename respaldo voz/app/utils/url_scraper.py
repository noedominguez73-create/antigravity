import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class URLScraper:
    MAX_CHARS = 5000
    TIMEOUT = 10  # seconds
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    @staticmethod
    def scrape(url):
        """
        Fetches the URL and extracts the main text content.
        Returns a dictionary with 'title' and 'content', or None on failure.
        """
        try:
            response = requests.get(url, headers=URLScraper.HEADERS, timeout=URLScraper.TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Get title
            title = soup.title.string if soup.title else url
            
            # Get text
            text = soup.get_text(separator='\n')
            
            # Break into lines and remove leading/trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > URLScraper.MAX_CHARS:
                text = text[:URLScraper.MAX_CHARS] + "... (truncated)"
                
            return {
                'title': title.strip(),
                'content': text
            }
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return None
