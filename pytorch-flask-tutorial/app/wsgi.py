# (should have heroku account)From terminal, app dir: heroku login -i 
#<enter email and pw>
# heroku create flask-pytorch-tutorial
# cd ..
# heroku local
#create runtime.txt
#on terminal: pip freeze > requirements.txt
#modify requirements.txt so that torch and torchvision version for CPU only (small enough so heroku will support) and Linux
#on terminal: git init, create .gitignore (search for python gitignore github, copy, add 'test/' at the first line)
# git gui
# heroku git:remote -a flask-pytorch-tutorial
# git add .
# git commit -m "initial commit"
# git push heroku master
#Copy url appearing when running then replace "http://localhost:5000" at test.py
from app.main import app
