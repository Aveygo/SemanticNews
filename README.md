# SemanticNews
![Rss](https://img.shields.io/badge/rss-F88900?style=for-the-badge&logo=rss&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)

<p align="center">
  <img src="https://raw.githubusercontent.com/Aveygo/SemanticNews/main/sample.png">
</p>

Sick of the "computer algorithms" that google news uses, I wanted to make a more open version of a news reader for personal use, with the benifit of zero tracking or cookies.

The theory is that, given a pool of article titles, major events would have similar titles and thus are headlines.

To compare headlines, semantics are extracted via distilbert-base-uncased, where we can use k-means to find the center of clusters and
rank headlines based on their distances.

Visit a version of it (running on a raspberry pi 3) [Here](https://semanticnews.dedyn.io:8080/)!

## RSS feed

An rss feed is supported at https://semanticnews.dedyn.io:8080/feed/rss as well as all other [endpoints](https://semanticnews.dedyn.io:8080/docs).

## Running

You will need python3 (tested on 3.9.2) and the following libraries

```
pip3 install fastapi uvicorn rfeed feedparser numpy transformers torch onnxruntime
```

Then just [download](https://github.com/Aveygo/SemanticNews/archive/refs/heads/main.zip), unzip, and start the local server and visit http://127.0.0.1:8080 on your browser!

```
python3 main.py
```

Note, the startup time will be awfully slow due to downloading and converting the bert model to onnx to run on a pi, 
as well as the initial population and vectorisation of articles.

## How you can help

Got any other rss source you want to see added? Chuck in a pull request for sources.py and Ill see to it.

Remember to give this repo a ⭐ if you found it useful.

[comment]: <> (tags: open source google news alternative)
[comment]: <> (tags: open news reader rss self hosted)
