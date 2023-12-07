# chatGSPP

- Web chatbot prototype tailored for GSPP's external website (gspp.berkeley.edu). Powered by GPT-3.5-Turbo.

- How to run the project:
        1. clone the repo:
        ```bash
        git clone https://github.com/yourusername/your-repo.git
        cd chatGSPP
        2. run the web-scraper
        ```bash
        python web-scraper.py
	3. run the question/answer system; adjust questions as needed
        ```bash
        python qa-system.py
	4. fine-tune with various hyperparameters and variables (more info below)
	5. embed the qa-system.py code within some fancy JS/HTML/CSS to make a UI-friendly chatbot

- Key hyperparameters and variables:
		- max_pages
		- max_len
		- max_tokens
		- input=x[:]
		- prompt
		- question
		- model
