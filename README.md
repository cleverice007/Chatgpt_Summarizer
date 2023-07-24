# Chatgpt_Summarizer

本專案旨在實現一個使用 OpenAI 的 Whisper 和 ChatGPT API 的應用。此應用能從 Podcast 中提取語音，並將其轉換成文字，再由 ChatGPT 提供摘要。用戶可以搜尋是否資料庫存有特定 Podcast 單集，並在找到對應的摘要後，進行與 ChatGPT 的互動。此專案同時實踐前後端設計、API 使用和 MongoDB 數據庫的練習。

# 功能
- 簡易搜尋: 允許用戶搜尋特定 Podcast 單集，檢查是否在資料庫中。
- 語音轉文字: 利用 OpenAI 的 Whisper API，將 Podcast 單集從 MP3 格式轉換為文字。
- 摘要生成: 利用 OpenAI 的 ChatGPT API，對轉錄的文字生成摘要。
- 對話功能: 根據存在資料庫的摘要，提供用戶與 ChatGPT 進行對話的功能。

# 使用技術
- 建立並配置 MongoDB 資料庫。
- 使用 YouTube API 獲取影片資訊並儲存到 MongoDB 中。
- 建立一個將影片資訊（如描述或字幕）送至 OpenAI GPT API 並獲取摘要的功能。
- 將產生的摘要與相對應的影片在 MongoDB 中進行關聯。
- 建立一個 API 用於搜尋 MongoDB 中的影片。
- 建立一個後端服務（使用 Express.js、Flask 或其他類似工具），該服務接收查詢請求，從 MongoDB 檢索影片，獲取其摘要，並返回結果。

