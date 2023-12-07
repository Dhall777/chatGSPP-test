## chatGSPP

- Web chatbot prototype tailored for GSPP's external website (gspp.berkeley.edu). Powered by GPT-3.5-Turbo.

- How to run the project:
        - pull the code
        - run the web-scraper
                - `python web-scraper.py`
	- run the question/answer system; adjust questions as needed
                - `python qa-system.py`
	- fine-tune with various hyperparameters and variables (more info below)
	- embed the qa-system.py code within some fancy JS/HTML/CSS to make a UI-friendly chatbot

- Key hyperparameters and variables:
		- `max_pages`
		- `max_len`
		- `max_tokens`
		- `input=x[:]`
		- `prompt`
		- `question`
		- `model`
