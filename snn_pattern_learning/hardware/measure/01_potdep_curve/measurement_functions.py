

def run_cycle_test(arduino):
    """첫 번째 실험: 전체 사이클 테스트를 수행합니다."""
    print("\n" + "="*50)
    print("### 🚀 실험 1: Cycle Test 시작 ###")
    print("="*50)

    all_cycle_data = []
    
    # 명령어 생성
    command_params = [
        "F", "5", "5", "GLOBAL_PD_SEQ_READ", "N56",
        "1", str(UPDATE_NUM), str(READ_PERIOD), "F",
        "5", "100", "100", "10", "10", "20", "10"
    ]
    command_to_send = ",".join(command_params)

    # SET_NUM 만큼 실험 반복
    for i in range(SET_NUM):
        print(f"\n--- Cycle {i + 1}/{SET_NUM} ---")
        print(f"➡️  명령어 전송: {command_to_send}")
        arduino.write((command_to_send + "\n").encode('utf-8'))

        print("⬅️  아두이노로부터 결과 수신 중...")
        current_cycle_rows = []
        while True:
            response = arduino.readline().decode('utf-8', 'ignore').strip()
            if response:
                if "EOD" in response:
                    print("--- Cycle 수신 완료 (EOD 수신) ---")
                    break
                if "Row_" in response:
                    data_part = response.split(',', 1)[1]
                    adc_values = [int(val) for val in data_part.replace('>', '').split(',')]
                    current_cycle_rows.append(adc_values)
            else:
                print("🔴 응답 시간 초과.")
                break
        
        if current_cycle_rows:
            all_cycle_data.append(current_cycle_rows)
            
    # 데이터 저장
    if all_cycle_data:
        current_time = datetime.datetime.now()
        filename = f"{current_time.strftime('%Y-%m-%d_%H-%M')}_CycleTest_Data.csv"
        header = [f'N5_ADC{i}' for i in range(5)] + [f'N6_ADC{i}' for i in range(5)]
        
        # 데이터를 1차원 리스트로 평탄화
        flat_data = [item for sublist in all_cycle_data for item in sublist]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(flat_data)
        print(f"\n✅ 실험 1 데이터 저장이 완료되었습니다. -> '{filename}'")
    
    print("\n### 🎉 실험 1: Cycle Test 종료 ###")


def perform_reset(arduino):
    """아두이노에 Reset 명령을 보내고 완료될 때까지 대기하는 함수"""
    print("리셋 명령을 생성합니다...")
    # Reset을 위한 명령어 파라미터 (10번의 리셋 펄스 인가)
    reset_params = [
        "F", "5", "5", "Reset", "N56",
        "1", "10", "10", "F",
        "5", "100", "100", "10",
        "10", "20", "10"
    ]
    reset_command = ",".join(reset_params) + "\n"

    print(f"➡️  리셋 명령어 전송: {reset_command.strip()}")
    arduino.write(reset_command.encode('utf-8'))

    print("⬅️  리셋 완료 신호(EOD) 수신 대기 중...")
    while True:
        response = arduino.readline().decode('utf-8', 'ignore').strip()
        if response:
            print(f"   수신: {response}")
            if "EOD" in response:
                print("--- 리셋 완료 ---")
                return True # 성공적으로 완료되면 True 반환
        else:
            print("🔴 리셋 중 타임아웃 발생.")
            return False # 실패 시 False 반환