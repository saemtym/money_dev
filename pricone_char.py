import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from urllib.parse import urljoin

def scrape_characters(url, save_dir):
    # HTMLを取得
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    characters = []
    for character_a in soup.find_all('a'):
        img_tag = character_a.find('img')
        if img_tag and 'i.png' in img_tag['src']:
            name = img_tag.get('alt')  # getメソッドを使用して、属性が存在しない場合にNoneを返すようにします

            if name and name != 'ダンジョン':  # nameがNoneでないかつ、nameが"ダンジョン"でない場合処理を行います
                img_url = urljoin(url, img_tag['src'])  # 相対URLを絶対URLに変換
                characters.append((name, img_url))

                # 画像をダウンロード
                response = requests.get(img_url)
                img = Image.open(BytesIO(response.content))
                img.save(os.path.join(save_dir, f"{name}.png"))
                # print(name, img_url)

# scrape_characters('https://gamewith.jp/pricone-re/article/show/92923', 'chars')

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# FLANNパラメータを設定
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

# FLANNマッチャを作成
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ORBディスクリプタを作成
orb = cv2.ORB_create()

# 大きな画像を読み込む
img1 = cv2.imread('decks/IMG_3309.png',0)  # queryImage
kp1, des1 = orb.detectAndCompute(img1, None)
des1 = np.float32(des1)
sift = cv2.SIFT_create()

# charsフォルダ内のすべての画像に対してマッチングを行う
for filename in os.listdir('chars'):
    img2 = cv2.imread(os.path.join('chars', filename),0) # trainImage

    # 画像の高さと幅を取得
    h, w = img2.shape[:2]

    # 画像の高さと幅の3%を計算
    h_crop = int(h * 0.03)
    w_crop = int(w * 0.03)

    # 画像をクロップ
    img2 = img2[h_crop:h-h_crop, w_crop:w-w_crop]

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Brute-Forceマッチャーを使用してマッチングを行う
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # マッチング結果をフィルタリング（デビッド・ローのアルゴリズムを使用）
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    # RANSACを使用してアウトライアーを除去
    if len(good_matches) > 4:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        if M is not None:
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            print("Match found in file: ", filename)
        else:
            print("Homography could not be computed.")
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches),4))
        matchesMask = None

    # マッチング結果を描画
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)

    # plt.imshow(img3, 'gray'),plt.show()
    # 描画したファイルを保存
    cv2.imwrite('matches.jpg', img3)
