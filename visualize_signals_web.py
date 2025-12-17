import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycaret.classification import load_model, predict_model
import jinja2
import os

# ==========================================
# AYARLAR
# ==========================================
FEATURE_FILE = '280_gunluk_feature_seti_.xlsx'
MODEL_V2_PATH = 'v2_experiment/fintech_v2_model'
MODEL_V3_PATH = 'v3_experiment/fintech_v3_model'
OUTPUT_HTML = 'Sinyal_Analiz_Raporu.html'
CONFIDENCE_THRESHOLD = 0.55

def apply_filter(signals, scores, threshold):
    sig_series = pd.Series(signals)
    score_series = pd.Series(scores)
    filtered_sig = np.where(score_series >= threshold, sig_series, np.nan)
    filtered_sig = pd.Series(filtered_sig)
    return filtered_sig.ffill().bfill().tolist()

def get_last_signal_info(dates, prices, signals):
    """
    Bir sinyal listesindeki son değişimi bulur.
    Döndürür: { 'type': 'AL'/'SAT', 'date': str, 'price': float, 'candles_ago': int, 'profit_pct': float }
    """
    if not signals or len(signals) == 0:
        return {'type': '-', 'date': '-', 'price': 0, 'candles_ago': 0, 'profit': 0}

    # Sondan başa doğru tara
    last_change_idx = 0
    
    # Sinyalin değiştiği son noktayı bul
    for i in range(len(signals) - 1, 0, -1):
        if signals[i] != signals[i-1]:
            last_change_idx = i
            break
            
    signal_val = signals[last_change_idx]
    signal_type = "AL" if signal_val == 1 else "SAT"
    signal_date = dates[last_change_idx]
    signal_price = prices[last_change_idx]
    current_price = prices[-1]
    candles_ago = (len(signals) - 1) - last_change_idx
    
    if signal_type == "AL":
        profit = ((current_price - signal_price) / signal_price) * 100
    else:
        profit = ((signal_price - current_price) / signal_price) * 100

    return {
        'type': signal_type,
        'date': signal_date,
        'price': signal_price,
        'candles_ago': candles_ago,
        'profit': profit
    }

def create_web_report():
    print("="*70)
    print("📊 PREMIUM SİNYAL RAPORU OLUŞTURULUYOR")
    print("="*70)

    print(f"📂 Veri okunuyor: {FEATURE_FILE}")
    try:
        df = pd.read_excel(FEATURE_FILE)
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.sort_values('DATE')
    except Exception as e:
        print(f"❌ Veri okuma hatası: {e}")
        return

    df['OPEN'] = df.groupby('CODE')['CLOSING_TL'].shift(1)
    df['OPEN'] = df['OPEN'].fillna(df['LOW_TL'])

    print("🧠 Modeller yükleniyor...")
    try:
        model_v2 = load_model(MODEL_V2_PATH)
        model_v3 = load_model(MODEL_V3_PATH)
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return

    print("🔮 Tahminler üretiliyor...")
    pred_v2 = predict_model(model_v2, data=df.copy(), verbose=False)
    pred_v3 = predict_model(model_v3, data=df.copy(), verbose=False)

    df['V2_Signal'] = pred_v2['prediction_label']
    df['V2_Score'] = pred_v2['prediction_score']
    df['V3_Signal'] = pred_v3['prediction_label']
    df['V3_Score'] = pred_v3['prediction_score']

    stocks_data = {}
    summary_data = [] # Tablo için özet veri
    unique_stocks = sorted(df['CODE'].unique())

    print(f"📈 İşlenen Hisse Sayısı: {len(unique_stocks)}")
    
    for stock in unique_stocks:
        stock_df = df[df['CODE'] == stock].copy()
        stock_df = stock_df.sort_values('DATE')
        
        dates = stock_df['DATE'].dt.strftime('%d-%m-%Y').tolist()
        prices = stock_df['CLOSING_TL'].tolist()
        
        # Sinyaller
        v2_filt = apply_filter(stock_df['V2_Signal'], stock_df['V2_Score'], CONFIDENCE_THRESHOLD)
        v3_filt = apply_filter(stock_df['V3_Signal'], stock_df['V3_Score'], CONFIDENCE_THRESHOLD)
        
        # GERCEKLESEN (Eski) veya Current_Trend (Yeni) kolonunu kullan
        if 'GERCEKLESEN' in stock_df.columns:
            real_sig = stock_df['GERCEKLESEN'].fillna(0).astype(int).tolist()
        elif 'Current_Trend' in stock_df.columns:
            real_sig = stock_df['Current_Trend'].fillna(0).astype(int).tolist()
        else:
            # Eğer ikisi de yoksa (örn. sadece tahmin için üretilmiş veride)
            real_sig = [0] * len(stock_df)
        
        # Grafik Verisi
        data = {
            'dates': dates,
            'open': stock_df['OPEN'].tolist(),
            'high': stock_df['HIGH_TL'].tolist(),
            'low': stock_df['LOW_TL'].tolist(),
            'close': prices,
            'v2_signals_filtered': v2_filt,
            'v3_signals_filtered': v3_filt,
            'actual_signals': real_sig
        }
        stocks_data[stock] = data
        
        # Özet Tablo Verisi
        info_v2 = get_last_signal_info(dates, prices, v2_filt)
        info_v3 = get_last_signal_info(dates, prices, v3_filt)
        info_real = get_last_signal_info(dates, prices, real_sig)
        
        summary_data.append({
            'stock': stock,
            'v2': info_v2,
            'v3': info_v3,
            'real': info_real
        })

    print("📝 HTML şablonu işleniyor...")
    
    template_str = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Sinyal Paneli</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0d1117;
            --panel-bg: #161b22;
            --text-main: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --border: #30363d;
            --success: #238636;
            --danger: #da3633;
            --buy-color: #3fb950;
            --sell-color: #f85149;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        
        /* Header */
        .header {
            background-color: var(--panel-bg);
            border-bottom: 1px solid var(--border);
            padding: 0 30px;
            height: 60px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .header h1 {
            margin: 0;
            font-size: 1.1rem;
            color: var(--text-main);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            height: 100%;
        }
        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 0 20px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            height: 100%;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab-btn:hover {
            color: var(--text-main);
            background-color: rgba(255,255,255,0.03);
        }
        .tab-btn.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }
        
        /* Content Area */
        .content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            position: relative;
        }
        .tab-pane {
            display: none;
            height: 100%;
            flex-direction: column;
        }
        .tab-pane.active {
            display: flex;
        }
        
        /* Chart Controls */
        .controls-bar {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            background: var(--panel-bg);
            padding: 15px;
            border-radius: 6px;
            border: 1px solid var(--border);
            align-items: center;
        }
        
        select {
            background-color: #0d1117;
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 6px;
            outline: none;
            font-family: inherit;
            min-width: 220px;
            font-size: 0.9rem;
        }
        select:focus {
            border-color: var(--accent);
        }
        label {
            font-size: 0.75rem;
            color: var(--text-muted);
            display: block;
            margin-bottom: 4px;
            font-weight: 600;
        }
        
        /* Chart */
        #chartDiv {
            flex: 1;
            background: var(--panel-bg);
            border-radius: 6px;
            border: 1px solid var(--border);
            min-height: 0; /* Flex için önemli */
        }
        
        /* Table */
        .table-container {
            background: var(--panel-bg);
            border-radius: 6px;
            border: 1px solid var(--border);
            overflow: auto;
            height: 100%;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            white-space: nowrap;
        }
        thead {
            position: sticky;
            top: 0;
            z-index: 10;
            background-color: #21262d;
        }
        th {
            padding: 12px 15px;
            font-weight: 600;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
            font-size: 0.85rem;
            cursor: pointer;
            user-select: none;
        }
        th:hover {
            color: var(--text-main);
            background-color: #30363d;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
            color: var(--text-main);
        }
        tr:hover {
            background-color: #21262d;
        }
        
        /* Badges */
        .badge {
            padding: 4px 8px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
            min-width: 40px;
            text-align: center;
        }
        .bg-buy { background-color: rgba(63, 185, 80, 0.15); color: var(--buy-color); border: 1px solid rgba(63, 185, 80, 0.4); }
        .bg-sell { background-color: rgba(248, 81, 73, 0.15); color: var(--sell-color); border: 1px solid rgba(248, 81, 73, 0.4); }
        
        .profit-pos { color: var(--buy-color); font-weight: 600; }
        .profit-neg { color: var(--sell-color); font-weight: 600; }
        
        .group-header { 
            text-align: center;
            border-left: 1px solid var(--border);
            background-color: #21262d;
        }
        .border-left { 
            border-left: 1px solid var(--border);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--bg-color); }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 5px; border: 2px solid var(--bg-color); }
        ::-webkit-scrollbar-thumb:hover { background: #484f58; }

    </style>
</head>
<body>

    <!-- Header -->
    <div class="header">
        <h1>
            <span style="color: var(--accent);">⚡</span> AutoML Trader
        </h1>
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('chart')">Grafik Analizi</button>
            <button class="tab-btn" onclick="switchTab('table')">Sinyal Özeti</button>
        </div>
    </div>

    <div class="content">
        
        <!-- TAB 1: GRAFİK -->
        <div id="tab-chart" class="tab-pane active">
            <div class="controls-bar">
                <div>
                    <label>HİSSE SENEDİ</label>
                    <select id="stockSelect" onchange="updateChart()">
                        {% for stock in stocks %}
                        <option value="{{ stock }}">{{ stock }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label>ANALİZ MODU</label>
                    <select id="modelSelect" onchange="updateChart()">
                        <option value="COMPARE_FILTERED">Kıyaslamalı (Hepsi)</option>
                        <option value="V2_FILTERED">V2 Filtreli (>%55)</option>
                        <option value="V3_FILTERED">V3 Filtreli (>%55)</option>
                        <option value="REAL">Gerçekleşen (İdeal)</option>
                    </select>
                </div>
            </div>
            <div id="chartDiv"></div>
        </div>

        <!-- TAB 2: TABLO -->
        <div id="tab-table" class="tab-pane">
            <div class="table-container">
                <table id="summaryTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)" style="width: 100px;">Hisse</th>
                            
                            <!-- V2 -->
                            <th class="group-header" colspan="4">V2 Filtreli (3 Günlük)</th>
                            
                            <!-- V3 -->
                            <th class="group-header" colspan="4">V3 Filtreli (5 Günlük)</th>
                            
                            <!-- Real -->
                            <th class="group-header" colspan="3">Gerçekleşen</th>
                        </tr>
                        <tr>
                            <th>Kod</th>
                            
                            <!-- V2 Cols -->
                            <th class="border-left" onclick="sortTable(1)">Son Sinyal</th>
                            <th onclick="sortTable(2)">Tarih</th>
                            <th onclick="sortTable(3)">Mum Önce</th>
                            <th onclick="sortTable(4)">Kâr/Zarar</th>
                            
                            <!-- V3 Cols -->
                            <th class="border-left" onclick="sortTable(5)">Son Sinyal</th>
                            <th onclick="sortTable(6)">Tarih</th>
                            <th onclick="sortTable(7)">Mum Önce</th>
                            <th onclick="sortTable(8)">Kâr/Zarar</th>
                            
                            <!-- Real Cols -->
                            <th class="border-left" onclick="sortTable(9)">Son Sinyal</th>
                            <th onclick="sortTable(10)">Tarih</th>
                            <th onclick="sortTable(11)">Mum Önce</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in summary %}
                        <tr>
                            <td style="font-weight: 700; color: var(--text-main);">{{ row.stock }}</td>
                            
                            <!-- V2 -->
                            <td class="border-left">
                                <span class="badge {{ 'bg-buy' if row.v2.type == 'AL' else 'bg-sell' }}">
                                    {{ row.v2.type }}
                                </span>
                            </td>
                            <td>{{ row.v2.date }}</td>
                            <td>{{ row.v2.candles_ago }}</td>
                            <td class="{{ 'profit-pos' if row.v2.profit > 0 else 'profit-neg' }}">
                                %{{ "%.2f"|format(row.v2.profit) }}
                            </td>

                            <!-- V3 -->
                            <td class="border-left">
                                <span class="badge {{ 'bg-buy' if row.v3.type == 'AL' else 'bg-sell' }}">
                                    {{ row.v3.type }}
                                </span>
                            </td>
                            <td>{{ row.v3.date }}</td>
                            <td>{{ row.v3.candles_ago }}</td>
                            <td class="{{ 'profit-pos' if row.v3.profit > 0 else 'profit-neg' }}">
                                %{{ "%.2f"|format(row.v3.profit) }}
                            </td>
                            
                            <!-- Real -->
                            <td class="border-left">
                                <span class="badge {{ 'bg-buy' if row.real.type == 'AL' else 'bg-sell' }}">
                                    {{ row.real.type }}
                                </span>
                            </td>
                            <td>{{ row.real.date }}</td>
                            <td>{{ row.real.candles_ago }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

<script>
    const stocksData = {{ stocks_data | tojson }};
    
    function switchTab(tabId) {
        // Butonları güncelle
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        event.target.classList.add('active');
        
        // Panelleri güncelle
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        document.getElementById('tab-' + tabId).classList.add('active');
        
        if(tabId === 'chart') {
            setTimeout(() => {
                Plotly.Plots.resize(document.getElementById('chartDiv'));
            }, 50);
        }
    }

    function getSignalTraces(dates, prices_high, prices_low, signals, name, colorBuy, colorSell, yAxis='y') {
        const buyX = [], buyY = [];
        const sellX = [], sellY = [];
        let prevSignal = null;
        
        for (let i = 0; i < dates.length; i++) {
            const currentSignal = signals[i];
            if (prevSignal !== null && currentSignal !== prevSignal) {
                if (currentSignal === 1) { 
                    buyX.push(dates[i]); buyY.push(prices_low[i] * 0.97); 
                } else { 
                    sellX.push(dates[i]); sellY.push(prices_high[i] * 1.03); 
                }
            }
            prevSignal = currentSignal;
        }
        
        return [
            { x: buyX, y: buyY, mode: 'markers', type: 'scatter', name: name + ' AL', marker: { symbol: 'triangle-up', color: colorBuy, size: 12, line: {width: 1, color: 'white'} }, yaxis: yAxis, hoverinfo: 'x+name' },
            { x: sellX, y: sellY, mode: 'markers', type: 'scatter', name: name + ' SAT', marker: { symbol: 'triangle-down', color: colorSell, size: 12, line: {width: 1, color: 'white'} }, yaxis: yAxis, hoverinfo: 'x+name' }
        ];
    }

    function updateChart() {
        const selectedStock = document.getElementById('stockSelect').value;
        const selectedModel = document.getElementById('modelSelect').value;
        const data = stocksData[selectedStock];
        
        if (!data) return;

        const traceCandle = {
            x: data.dates,
            open: data.open, high: data.high, low: data.low, close: data.close,
            type: 'candlestick', name: 'Fiyat',
            increasing: {line: {color: '#3fb950', width: 1}, fillcolor: '#3fb950'},
            decreasing: {line: {color: '#da3633', width: 1}, fillcolor: '#da3633'},
            hoverlabel: { bgcolor: '#161b22' }
        };

        let plotData = [];
        let layout = {};

        const commonLayout = {
            plot_bgcolor: '#0d1117',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9', family: 'Inter, sans-serif' },
            xaxis: { type: 'category', rangeslider: { visible: false }, gridcolor: '#21262d', showline: false, tickmode: 'auto', nticks: 10, spikemode: 'across', showspikes: true },
            yaxis: { gridcolor: '#21262d', tickformat: '.2f', spikemode: 'across', showspikes: true },
            dragmode: 'pan',
            hovermode: 'x',
            margin: { t: 50, b: 40, l: 60, r: 40 },
            legend: { orientation: 'h', y: 1.02, x: 0.5, xanchor: 'center', font: {size: 10} }
        };

        if (selectedModel === 'COMPARE_FILTERED') {
            layout = {
                ...commonLayout,
                grid: {rows: 3, columns: 1, pattern: 'independent', roworder: 'top to bottom'},
                yaxis: { ...commonLayout.yaxis, domain: [0.68, 1], title: 'V2 Filtreli' },
                yaxis2: { ...commonLayout.yaxis, domain: [0.34, 0.66], title: 'V3 Filtreli' },
                yaxis3: { ...commonLayout.yaxis, domain: [0, 0.32], title: 'GERÇEKLEŞEN' },
                xaxis: { ...commonLayout.xaxis, anchor: 'y3' },
                height: 900,
                title: `${selectedStock} - 3'lü Karşılaştırma`
            };

            plotData = [
                traceCandle, 
                ...getSignalTraces(data.dates, data.high, data.low, data.v2_signals_filtered, 'V2', '#3fb950', '#da3633', 'y'),
                {...traceCandle, yaxis: 'y2', showlegend: false},
                ...getSignalTraces(data.dates, data.high, data.low, data.v3_signals_filtered, 'V3', '#58a6ff', '#f85149', 'y2'),
                {...traceCandle, yaxis: 'y3', showlegend: false},
                ...getSignalTraces(data.dates, data.high, data.low, data.actual_signals, 'GERÇEK', '#e3b341', '#d29922', 'y3')
            ];
        } else {
            let signals, name, colorBuy, colorSell;
            if (selectedModel === 'V2_FILTERED') { signals = data.v2_signals_filtered; name = 'V2 Filtreli'; colorBuy = '#3fb950'; colorSell = '#da3633'; }
            else if (selectedModel === 'V3_FILTERED') { signals = data.v3_signals_filtered; name = 'V3 Filtreli'; colorBuy = '#58a6ff'; colorSell = '#f85149'; }
            else { signals = data.actual_signals; name = 'GERÇEK'; colorBuy = '#e3b341'; colorSell = '#d29922'; }

            layout = { ...commonLayout, title: `${selectedStock} - ${name}`, height: 700 };
            plotData = [traceCandle, ...getSignalTraces(data.dates, data.high, data.low, signals, name, colorBuy, colorSell, 'y')];
        }

        Plotly.newPlot('chartDiv', plotData, layout, {responsive: true, scrollZoom: true, displayModeBar: false});
    }

    function sortTable(n) {
        var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
        table = document.getElementById("summaryTable");
        switching = true;
        dir = "asc"; 
        while (switching) {
            switching = false;
            rows = table.rows;
            for (i = 2; i < (rows.length - 1); i++) {
                shouldSwitch = false;
                x = rows[i].getElementsByTagName("TD")[n];
                y = rows[i + 1].getElementsByTagName("TD")[n];
                
                // Sayısal kontrol
                var xContent = x.innerText.replace(/%|\s/g, '');
                var yContent = y.innerText.replace(/%|\s/g, '');
                var isNumeric = !isNaN(parseFloat(xContent)) && !isNaN(parseFloat(yContent));
                
                var cmpX = isNumeric ? parseFloat(xContent) : x.innerHTML.toLowerCase();
                var cmpY = isNumeric ? parseFloat(yContent) : y.innerHTML.toLowerCase();

                if (dir == "asc") {
                    if (cmpX > cmpY) { shouldSwitch = true; break; }
                } else if (dir == "desc") {
                    if (cmpX < cmpY) { shouldSwitch = true; break; }
                }
            }
            if (shouldSwitch) {
                rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                switching = true;
                switchcount ++;      
            } else {
                if (switchcount == 0 && dir == "asc") { dir = "desc"; switching = true; }
            }
        }
    }

    window.onload = updateChart;
</script>
</body>
</html>
    """
    
    template = jinja2.Template(template_str)
    html_content = template.render(stocks=unique_stocks, stocks_data=stocks_data, summary=summary_data)
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"✅ PREMIUM Rapor oluşturuldu: {OUTPUT_HTML}")

if __name__ == "__main__":
    create_web_report()
