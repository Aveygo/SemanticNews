import sqlite3, base64, zlib, base64, numpy as np, datetime
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from rfeed import *

from articles import Downloader

d = Downloader(reoccuring=True)
d.start()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def mse(x, y):
    try:
        x = d.decode(x)
        y = d.decode(y)
        return 1 / (1 + np.mean(x-y) ** 2)
    except:
        return 1000

#conn = sqlite3.connect('news.db', check_same_thread=False)
#c = conn.cursor()
d.c.execute('''CREATE TABLE IF NOT EXISTS news (news_id INTEGER PRIMARY KEY AUTOINCREMENT, title text, link text, summary text, published text, source text, source_name text, media text, compvec text)''')
d.conn.create_function("mse", 2, lambda x, y: mse(x, y))

@app.get("/")
async def root():
    """
    Index page
    """
    return HTMLResponse(open("index.html").read())

@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.png")

@app.get("/feed/headlines")
async def get_headlines(amount: int=10, offset: int=0) -> list[dict]:
    """
    Get the latest headlines, updates every hour - time to download/process
    """
    headlines = d.cache["headlines"]
    return headlines[offset:offset+amount]

@app.get("/feed/latest_added")
async def get_latest(amount: int=10, offset: int=0) -> list[dict]:
    """
    Get the latest articles added to the database
    """
    d.c.execute("SELECT * FROM news ORDER BY news_id DESC LIMIT ? OFFSET ?", (amount, offset))
    return [d.parse_row(row) for row in d.c.fetchall()]

@app.get("/feed/latest_published")
async def get_latest(amount: int=10, offset: int=0) -> list[dict]:
    """
    Get the latest articles published
    """
    d.c.execute("SELECT * FROM news ORDER BY published DESC LIMIT ? OFFSET ?", (amount, offset))
    return [d.parse_row(row) for row in d.c.fetchall()]

@app.get("/feed/similar")
async def get_similar(compvec: str, amount: int=10, offset: int=0) -> list[dict]:
    """
    Get the most similar articles to the given compvec
    """
    try:
        np.frombuffer(zlib.decompress(base64.urlsafe_b64decode(compvec)), dtype=np.float16)
    except:
        raise HTTPException(400, detail="Invalid compvec, could not load")

    d.c.execute("SELECT * FROM news ORDER BY mse(compvec, ?) DESC LIMIT ? OFFSET ?", (compvec, amount, offset))
    return [d.parse_row(row) for row in d.c.fetchall()]

@app.get("/feed/random")
async def get_random(amount: int=10) -> list[dict]:
    """
    Get random articles
    """
    d.c.execute("SELECT * FROM news ORDER BY RANDOM() LIMIT ?", (amount,))
    return [d.parse_row(row) for row in d.c.fetchall()]

@app.get("/feed/search")
async def search(query: str, source:str=None, amount: int=10, offset: int=0) -> list[dict]:
    """
    Search for articles
    """
    if source is None:
        d.c.execute("SELECT * FROM news WHERE title LIKE ? ORDER BY news_id DESC LIMIT ? OFFSET ?", (f"%{query}%", amount, offset))
    else:
        d.c.execute("SELECT * FROM news WHERE title LIKE ? AND source_name = ? ORDER BY news_id DESC LIMIT ? OFFSET ?", (f"%{query}%", source, amount, offset))
    
    return [d.parse_row(row) for row in d.c.fetchall()]

@app.get("/feed/sources")
async def get_sources() -> list[str]:
    """
    Get a list of all sources used in the database
    """
    d.c.execute("SELECT DISTINCT source_name FROM news")
    return [row[0] for row in d.c.fetchall()]

@app.get("/feed/rss")
async def get_rss(page: int=0) -> PlainTextResponse:
    """
    Get an RSS feed of the latest headlines
    """
    if page > 10:
        raise HTTPException(400, detail="Page number too high")

    headlines = d.cache["headlines"][page*50:(page+1)*50]
    
    items = []
    for article in headlines:
        items.append(Item(
            title=article["title"],
            link=article["link"],
            description=article["summary"],
            pubDate=datetime.datetime.fromtimestamp(int(article["published"])),
            guid = Guid(article["link"]),
            author=article["source_name"]
        ))

    feed = Feed(
        title="Semantic Headlines",
        link="https://github.com/Aveygo",
        description="Using semantic search to find the most relevant headlines from 40 sources, updated hourly.",
        language="en-US",
        lastBuildDate=datetime.datetime.now(),
        items=items
    )

    return PlainTextResponse(feed.rss(), media_type="text/xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    # uvicorn.run(app, host="0.0.0.0", port=443,ssl_keyfile="/etc/letsencrypt/live/semanticnews.dedyn.io/privkey.pem", ssl_certfile="/etc/letsencrypt/live/semanticnews.dedyn.io/fullchain.pem")

"""
x = "eJwN1GlYVWUCwHHcVxCV7V7Ovffs5z3nXgRRRCtFScF9VMRBTTOXUSkUFXeUyhVEAVmEu57lfc95z7mXSie1nkwnrBB1Ksc9DU0bmdQaG8dySGv6-v_8e_6M2C-Yahznbtkm0vvxJmEL96PxHn9FfMNoN_-SstkZb4bYDqaupkuElA6a4wClZFKq_hh18H9Ti-hfQs_1FjozWE1myVp4TfJWV2_zReYusY-qwon-PKtV_1adhz_Rp9ZMEkoABFqoFL4idY_cDqzWt3sn4_epWC2NuMWV-NLMBNcO9rDVSQTwwiDJuFFfLlOfVTEOHRQPi9uovNA5tg2t8Dr4lwwrXGaeGxxjtEqMOUN4BG_jiUJUME9xcRcjKp7HLwKnIm7yZNJV_oHXCd-ks1MuCZ9iRSiTP2PHiNvgDrDH9AaqzV31j-Wo4LjmC-LoxvmeM4gx_yT44HhxQtxX6jDxn0lxxi2FoBU4ih2p3RZOGB-Ke6WdyZNwo1QcgfoX4ttNAWW5EoLvOgrpBF7oUipcpxh0lu-nlJrx_I1kmzbPM9czV8iRpvuHGE3wmCRa37oK5cHgH7iPFZIY_TFtsO9aq00bzDGHgL3BQouMHyQ7Yb5tVNM46rI_bHbFSridfaFxIMx0jo7cU5Po3-NbjHrHSRGjhU1xqN2RQa4Vs_VR7iT2dWiEchQnzBS7uOdYz8wroYFuQTuhjTDb3aY_13NFTFKLEeNcUl_ZFJvUQtkbRjqsiBtcEXZ4Zrtfpn9n95nb3elme32nxTeFRbf5LadTGf6jphY5IxbCSXCXOA1-z4-IdEMf-1rUppR8IY7dLRW4d-gQ50UWeJ7LRY6MYCTil4Z7NkTWSaSaanSAflSx0ZBZi7szcTCE13KLrQSpxazCM5nlHGHtI6ahLHkBuYxRGTZxtpLINQbqyJgw4Ykho3GaOpLralXG1qrnhRt8DCwzBH816s_Oc28DrwirTVlYDE9y--zLlD6OpNAHHjsjoRHEeSORa4YJnoXiWfAxzhVHgXycHWk0XyFXOuL4doFJ6GiYo2zUHdpVYZERDfpDnRuq2xoLa-_rlFKub1GS4mO5GFwvFTCH605UfSAM4I7KHN4nlJuHcE4d9KzRDnE70EOmlJqh9sD5npn4XdxqBJyB8FlUwBL0Y31tykVprLsDO-RhoBYPxOuMGOpStNY0AHbBnRLwr3avb-7h3MGs8LwzKDgYIkHf4zzlEgPN6jvUR9JR3GzyHEvN5xT1Kj0FLk2ZEBxvpjYuSrmFxpNzlFn4pHFIqeKr-AMwA15GnaC7sSW5P54ppfuHaMXNOw0GvSxv4TcbbcQuoQ09dbeblNXh8qMoU4AlsIhqwvHN96JPaO85iqk0VIaOsm-aLzkuJIDYFDyKe2Qcr-5JPdE70XZcLW1k3aTFgsBcPCvQoRuUDhu17oEsbGljtAvSYKsDpHEEUUa9J_0Wl2-N0H_G3ygnjAr5HGysHalPEzLcbVSl_DPdgreQx7RnwkJuOT0L_t03Sa0XfiKe0vvEHCS6Ct3n40fxF_UJuGskBUwRsgHB7Ne6oOG8g6_gBKnKZDREIWaz_2f8KzXGNdBcqnTzXOU65XaRcGTp7ws_1cXiGlil95Q3wAopD95mlwJL_S_5IHYe-oQp1H6hJhs58lAu2hoIviQJa_yQYQzPV9U_1EepXzotvqcYQWl-jR9WPwLu4-vcNN6pdxj_cY2P64b_JDKBJ-wB_jgcxH9HfQjng4WOA6YUeYEstNNcCnhbmmrtSDU9M5hD5evd7wT2k6elQhjNxsDpuC-YzCTaLrp3-XXHGthPJtB97zuVgrMUvqAni3l7f5BrfDlwnt4qBM0pVJLgps7TfW0PmY36Tbk3paG3lGf-TvZ_BMX_wJ4WJyUfIb8A2HOl7ke-xGpg3w9qdDG-SJ9Cq4wtNBnJoC-gHtZo8B39FDrCiFqgZSSsDPUGJ-ENJV3bIDcrRcRY5253yD1IEqAnuEkeGJzty7YBua3uijGaLggBuTt7iRodvubNkjxoifYcHKfz667z3QPJQm9YS3UFyh_v7h_wwa-IO445YIQxHLzt_Bc1ETyL8qpn5SPGFfdyWgAu_8tUb7ZcHub6t5GnIvV08LHeik-6hvNBVXQ6Gw1jLsgOEYq3lyys0nrpOtdpHJQukC_i66hNa0ExZgU7g9wWughHKlXkI9c46s-BMLslpp8YBacmtVvxdLY8BWxXSrEm_pS4lW4BPYVcZYwazd8MHqZ_Q0XR8Xwlnmi9qu1nVzoP2UfC6Q5Om21Mk067-rI3pYyaXqSb_UxfWft37EqeTC0RupG7UE-YPWim-av-uXCLuuy8HnquCWwN9IGjkAhU-6IED0wAWWJb3Hb-Pr2eewofkgV8KrKLpHxHoISw0tMxlD3Pr9EKnZfx7-GpDJesgRb1iD66JlaQ-PlMsnku1st3Nb-g21GOOZ-YYFS7boCgnk6nhu77J9nymd3cAjZi62rsEtqdZeA4Sg2W6inJ45NrWaz_UbQB_gS4ij7iyA08VDPrDdzbPcN4yf8a-NK-tbGM3Rv6GF0OcGKac79RTGxFa0UHmswn6DHoyoB0ysYssjVRs9hPoA3e9L_BVqqvawdNSl1guMVv-YmaGB_PtpOLlEyzB4jC1-1d7LuZKDaEP4HdfKRyROi7r1_oEZWmDRA_Mkrw50SCcT78I8QqJx0yCW8Xfw6b674VmYl-c5HqDGIdfqqvdfby_TzqCX1VzWMvERP5CuZT5bTgkj0cbU4VFyHMOb3PgB1fS8rwX-PW4ou-Vep58XUr0Z_gsPGr41Pd6UwRqItrxY9cUUp24j3uAPk3Y29wspwIpoM5wnCyiDmjLiO2sh_BxeAcLFXLmHR0yT7DeqDfYG1URdNVtsCxnjoLGUehftv7jfEa8KrNXAGcCyn_VTYan9XT1ZXU6YZc6qHYAG4yxdwm3f6HldPwvjhGGQ2fmFPsX6suQdeHoo-oSnUnLhk0BavyZPV9uIHN4krNr_hK-i7T4P9RKBJjqWnKVuE1XIcvo6A0EsxUunp9vjSq3PtAOKXIXJ7ypR4DPkj6hYupL2tYZUZp042NZB0_Rs5iD9sbhFf1QjkbmUIT9nIX1FbSxreYW3Cr-L_qa31fDJ-yRctfs7mhqdwBvhse63vSp5VZQTRrU4XKeClR0Tfp5WqYu0f2IL7RlgwmxIfw9qCW-kd4LJlMXYPdE2Tbd4FyvqBxNkiTfkUbnDGcyGQlPNCWBbJccewex134hLXzK8mxoXxk0E-VvmBFsFdo4cE6VKI8ESN4s3YHX1D7Gvm4i_sz7r-2_c17gM3qkJf7AfNXrJjV4AjxFj-f-4_rN-JDZpxR9mYJ7mZ6za9Rpf04KKHWNC4m4pjfjf1KijIZ9FHWQz4k4rmhKfgqnkCsZXsZn3NukGfdCe7m97AtDa-yKeF1fJX5gSFJYbsA3wiUoAniVrYg9IL6D_AKXkrNAWdgGhxIfwxZ-Se-qH4Mn1G9iLzXwKqvB14yirlmYzs3nVtY82LyD9DFfQqy5Ae8jH5xzISG5Ut-Bu_ZGSdHdndslqvxc7AJ7GWPsYeD_YxxEcjuBsc0N7_avw7Y0c36M7FL6AC72NlVK4-7CR_Tl-C_PHfQp2h-012cadhBUfg7M91Ywb0WyJRnK0V6LRuHHwdv9lptbLQO8HfQVKWNGc1-z_yVHVz5PfuG_5sgxYcbc0EqXsaPI0rIg-x15mRdSmUbF2t0anflCdJc705qmVROfGEv0noH7ajNFR3m1Q8PIRSMrBcr_LfEp7ZlagTS7BiYI1BSLtEB3sPbKhLlGfGLcKs-xDZUWyp_Fd-HzEM9QT9fLjjD9udfxZ85jlo15jHj_15RJCA="
x = np.frombuffer(zlib.decompress(base64.urlsafe_b64decode(x)), dtype=np.float16)
[x, y] = np.dot(sample, x)
"""
