# TDA Market Monitor

Сайт: вводите индекс → получаете Wasserstein Distance и нормы L1/L2
persistence landscapes для H0 и H1 с порогами μ+1σ … μ+4σ.

## Файлы
- `app.py` — интерфейс (Streamlit + Plotly)
- `tda_core.py` — загрузка данных и все расчёты
- `requirements.txt` — зависимости

## Вариант 1. Постоянный сайт (бесплатно, Streamlit Community Cloud)
1. Создайте репозиторий на GitHub и загрузите туда три файла.
2. Зайдите на https://share.streamlit.io → «Create app» → выберите репозиторий,
   main file = `app.py` → Deploy.
3. Получите ссылку вида `https://<имя>.streamlit.app`.

## Вариант 2. На своём компьютере
```bash
pip install -r requirements.txt
streamlit run app.py
```
Откроется http://localhost:8501

## Вариант 3. Google Colab (временная ссылка)
Загрузите три файла в /content и выполните в ячейке:
```python
!pip install -q -r requirements.txt
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
!nohup streamlit run app.py --server.port 8501 --server.headless true > st.log 2>&1 &
!./cloudflared tunnel --url http://localhost:8501
```
В выводе появится ссылка `https://….trycloudflare.com`.

## Что вводить
- WSJ: `RUT`, `SPX`, `DJIA`, `COMP`; для зарубежных — путь как в адресе WSJ,
  например `HK/XHKG/HSI`, или просто вставьте полную ссылку на страницу Historical Prices.
- Yahoo: `^RUT`, `^GSPC`, `^STOXX50E`, `^N225` …
- Несколько через запятую (`SPX, DJIA, COMP, RUT`) → многомерное облако точек
  (как в вашем коде для 5 индексов OMX Baltic).
- Один индекс → вложение Такенса (d=4, τ=1 по умолчанию).
- Baltic-индексов нет ни на WSJ, ни на Yahoo — выберите «Загрузить файлы»
  и загрузите ваши xlsx с Nasdaq Baltic.

## Методика (совпадает с вашими скриптами)
- Лог-доходности r_t = ln(P_t / P_{t−1}), окно w (по умолчанию 50).
- Диаграммы: Vietoris–Rips (ripser), бесконечная точка H0 отбрасывается.
- Wasserstein: окно [t−2w+1, t−w] против [t−w+1, t], формула как в persim.wasserstein.
- Нормы: ||λ||_p = (Σ_k ∫ λ_k^p)^{1/p}, точный расчёт (как PersLandscapeExact.p_norm).
- Пороги: μ + kσ по всей выборке или только по прошлым данным.
