import os
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
threads = 4
timeout = 120