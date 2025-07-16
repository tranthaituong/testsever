import os
import requests
import networkx as nx
import matplotlib.pyplot as plt
import asyncio
import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from typing import List, Dict, Any, Optional

from pyvis.network import Network
import webbrowser
import os
# --- CẤU HÌNH ---
load_dotenv()

# URL API của bạn (đang chạy cục bộ)
FRAUD_API_URL = "http://127.0.0.1:8000/analyze"

# URL và API Key của Etherscan
ETHERSCAN_API_URL = "https://api.etherscan.io/api"
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

# Ngưỡng xác suất để phân loại màu sắc
PROB_FRAUD_THRESHOLD = 0.6  # Trên mức này là Đỏ (Gian lận)
PROB_SUSPICIOUS_THRESHOLD = 0.4  # Trên mức này là Vàng (Nghi ngờ)


# Dưới mức này sẽ là Xanh (An toàn)

# --- CÁC HÀM XỬ LÝ ---

async def get_fraud_prediction(session: aiohttp.ClientSession, address: str) -> Optional[Dict[str, Any]]:
    """Gọi API dự đoán cục bộ một cách bất đồng bộ để lấy kết quả phân tích."""
    payload = {"address": address}
    try:
        async with session.post(FRAUD_API_URL, json=payload, timeout=60) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Lỗi khi dự đoán địa chỉ {address[:10]}...: Status {response.status}")
                return None
    except Exception as e:
        print(f"Không thể kết nối đến API dự đoán cho địa chỉ {address[:10]}...: {e}")
        return None


def get_transactions(address: str) -> List[Dict[str, Any]]:
    """Lấy danh sách các giao dịch từ Etherscan API."""
    print(f"\n🔍 Đang lấy giao dịch cho địa chỉ: {address}")
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "1":
            print(f"✅ Tìm thấy {len(data['result'])} giao dịch.")
            return data["result"]
        else:
            print(f"⚠️ Không tìm thấy giao dịch hoặc có lỗi từ API: {data['message']}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gọi Etherscan API: {e}")
        return []


def get_node_color(probability_fraud: float) -> str:
    """Xác định màu của node dựa trên xác suất gian lận."""
    if probability_fraud > PROB_FRAUD_THRESHOLD:
        return 'red'  # Gian lận
    elif probability_fraud > PROB_SUSPICIOUS_THRESHOLD:
        return 'orange'  # Nghi ngờ (màu vàng)
    elif probability_fraud >= 0:
        return 'green'  # An toàn
    else:
        return 'grey'  # Không xác định


# THAY THẾ TOÀN BỘ HÀM CŨ BẰNG HÀM NÀY

#
# THAY THẾ TOÀN BỘ HÀM CŨ BẰNG HÀM NÀY
#
def draw_transaction_graph(central_address: str, transactions: List[Dict[str, Any]], predictions: Dict[str, Dict]):
    """
    Vẽ biểu đồ mạng lưới giao dịch TƯƠNG TÁC bằng PyVis.
    - Click vào node để mở Etherscan.
    - Hover vào cạnh để xem giá trị giao dịch.
    - Hover vào node để xem 'status' do model trả về và xác suất.
    - Giao diện hiện đại và có thể tương tác.
    """
    print("\n🎨 Bắt đầu vẽ biểu đồ mạng lưới tương tác với PyVis...")

    net = Network(height="90vh", width="100%", bgcolor="#222222", font_color="white", directed=True)

    unique_addresses = set()
    for tx in transactions:
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        if from_addr: unique_addresses.add(from_addr)
        if to_addr: unique_addresses.add(to_addr)

    for address in unique_addresses:
        prediction = predictions.get(address, {})
        prob = prediction.get('probability_fraud', -1)
        color = get_node_color(prob)
        status_from_model = prediction.get('prediction', 'Unknown')
        short_label = f"{address[:6]}...{address[-4:]}"
        etherscan_url = f"https://etherscan.io/address/{address}"

        # ⭐ SỬA LỖI TẠI ĐÂY ⭐
        # Quyết định chuỗi hiển thị cho xác suất TRƯỚC KHI đưa vào f-string lớn.
        prob_display_string = f"{prob:.2%}" if prob >= 0 else "N/A"

        title_html = (
            f"<b>Address:</b> {address}<br>"
            f"<b>Model Status:</b> <b style='color: {color};'>{status_from_model.capitalize()}</b><br>"
            # Sử dụng chuỗi đã được tạo sẵn ở trên
            f"<b>Fraud Probability:</b> {prob_display_string}<br><br>"
            f"<a href='{etherscan_url}' target='_blank' style='color: #87CEEB;'>Click to view on Etherscan</a>"
        )

        node_size = 25 if address == central_address else 15
        net.add_node(address, label=short_label, size=node_size, color=color, title=title_html)

    for tx in transactions:
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        if not from_addr or not to_addr:
            continue
        try:
            value_wei = int(tx.get('value', 0))
            value_eth = value_wei / 1e18
            edge_title = f"Value: {value_eth:.6f} ETH"
            net.add_edge(source=from_addr, to=to_addr, title=edge_title, value=value_eth * 0.1 + 1)
        except (ValueError, TypeError):
            net.add_edge(source=from_addr, to=to_addr, title="Invalid value")

    net.set_options("""
    const options = {
      "nodes": {
        "font": {
          "size": 12,
          "face": "Tahoma"
        }
      },
      "edges": {
        "color": {"inherit": "from"},
        "smooth": {
          "type": "dynamic",
          "forceDirection": "none",
          "roundness": 0.5
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1.4
          }
        },
        "arrowStrikethrough": false
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 230,
          "springConstant": 0.08,
          "avoidOverlap": 0.5
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "tooltipDelay": 200
      }
    }
    """)

    file_name = f"transaction_network_{central_address[:8]}.html"
    try:
        net.save_graph(file_name)
        print(f"\n✅ Biểu đồ tương tác đã được lưu vào tệp: {file_name}")
        webbrowser.open(f'file://{os.path.realpath(file_name)}')
    except Exception as e:
        print(f"Lỗi khi lưu hoặc mở biểu đồ: {e}")


async def main():
    """Hàm chính điều phối toàn bộ quy trình."""
    if not ETHERSCAN_API_KEY:
        print("LỖI: Biến môi trường ETHERSCAN_API_KEY chưa được thiết lập.")
        print("Vui lòng tạo file .env và thêm key vào đó.")
        return

    print("--- Trình phân tích và trực quan hóa mạng lưới giao dịch Ethereum ---")

    # 1. Nhập địa chỉ ví từ người dùng
    central_address = input("Nhập địa chỉ ví Ethereum bạn muốn phân tích: ").strip().lower()

    # 2. Lấy danh sách giao dịch từ Etherscan
    transactions = get_transactions(central_address)
    if not transactions:
        return

    # 3. Thu thập tất cả các địa chỉ duy nhất
    unique_addresses = {central_address}
    for tx in transactions:
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        if from_addr:
            unique_addresses.add(from_addr)
        if to_addr:
            unique_addresses.add(to_addr)

    print(f"\n🔬 Tìm thấy {len(unique_addresses)} địa chỉ duy nhất. Bắt đầu dự đoán...")

    # 4. Gọi API dự đoán cho tất cả các địa chỉ (bất đồng bộ để tăng tốc)
    predictions = {}
    async with aiohttp.ClientSession() as session:
        tasks = [get_fraud_prediction(session, addr) for addr in unique_addresses]
        # Sử dụng tqdm để hiển thị thanh tiến trình
        results = await tqdm.gather(*tasks, desc="Đang dự đoán")

    for res in results:
        if res and 'address' in res:
            predictions[res['address'].lower()] = res

    # 5. Vẽ biểu đồ
    draw_transaction_graph(central_address, transactions, predictions)


if __name__ == "__main__":
    # Chạy vòng lặp sự kiện bất đồng bộ
    asyncio.run(main())