# Chatgpt_Summarizer
利用whisper、chatgpt 將podcast 進行分類，重點摘要，達到podcast 知識的快速吸收

- 建立並配置 MongoDB 資料庫。
- 使用 YouTube API 獲取影片資訊並儲存到 MongoDB 中。
- 建立一個將影片資訊（如描述或字幕）送至 OpenAI GPT API 並獲取摘要的功能。
- 將產生的摘要與相對應的影片在 MongoDB 中進行關聯。
- 建立一個 API 用於搜尋 MongoDB 中的影片。
- 建立一個後端服務（使用 Express.js、Flask 或其他類似工具），該服務接收查詢請求，從 MongoDB 檢索影片，獲取其摘要，並返回結果。

