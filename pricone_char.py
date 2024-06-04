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

# テンプレートマッチングを行う画像を読み込むクラス
    

match_results = {}

# テンプレートマッチングを行う画像を読み込む
img = cv2.imread('decks/IMG_3309.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭を囲む矩形を取得
rects = [cv2.boundingRect(contour) for contour in contours]

# 矩形を描画
for i, rect in enumerate(rects):
    x, y, w, h = rect
    aspect_ratio = w / h
    if 0.9 <= aspect_ratio <= 1.1 and w >= 90 and h >= 90:  # 正方形に近く、72x72ピクセル以上の形状のみを描画
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi = img[y:y+h, x:x+w]
        # 6/96をかけた値を四隅から除外
        exclude_val = int(7/96 * min(w, h))
        roi = roi[exclude_val:-exclude_val, exclude_val:-exclude_val]

        best_match_value = -1
        best_match_file = None
        
        # パターンマッチングを行う

        
        for filename in os.listdir('chars'):
            template = cv2.imread(os.path.join('chars', filename), cv2.IMREAD_COLOR)

            # テンプレート画像の4つの端から7ピクセルを除外する
            h, w, _ = template.shape
            template = template[7:h-7, 7:w-7]

            res = cv2.matchTemplate(roi,template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            # マッチ度を保存
            match_results[i] = max_val

            if max_val > best_match_value:
                best_match_value = max_val
                best_match_file = filename

        print(f"Best match for rectangle {i} is {best_match_file} with match value {best_match_value}")

# best_match_id = max(match_results, key=match_results.get)
# best_match_value = match_results[best_match_id]




# 結果を保存
cv2.imwrite('result.png', img)





# # printする
# for i, rect in enumerate(rects):
#     x, y, w, h = rect
#     print(f"rect{i}: x={x}, y={y}, w={w}, h={h}")

# results = []

# # charsフォルダ内の各画像に対してテンプレートマッチングを行う
# for filename in os.listdir('chars'):
#     template = cv2.imread(os.path.join('chars', filename), cv2.IMREAD_COLOR)

#     # テンプレート画像の4つの端から7ピクセルを除外する
#     h, w, _ = template.shape
#     template = template[7:h-7, 7:w-7]

#     # テンプレートマッチングを行う
#     result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#     # 一定の閾値以上のマッチング結果のみを出力する
#     threshold = 0.6
#     if max_val >= threshold or filename in ['エリス.png', 'キャル(プリンセス).png', 'ヴァンピィ.png', 'ジータ(ウォーロック).png', 'クウカ(ノワール).png']:
#         results.append(f"{filename[:-4]} : {max_val}\n")

# with open('pricone_char.txt', 'w') as f:
#     f.writelines(results)