import math
from datetime import datetime
import os
import ast

index = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5,
    6:6,
    7:7,
    8:8,
    9:9,
    10:10,
    11:11,
    12:12,
    13:13,
    14:14,
    15:15,
    16:16,
    17:17,
    18:18,
    19:19,
    20:20,
    21:21,
    22:22,
    23:23,
    24:24,
    25:25,
    26:26,
    27:27,
    28:28,
    29:29,
    30:30,
    31:31,
    32:32,
    33:33,
    34:34,
    35:35,
    36:36,
    37:37,
    38:38,
    39:39,
    40:40,
    41:41
}  # 관절번호 인덱스 치환

order = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
    (21, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (21, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (21, 34),
    (34, 35),
    (35, 36),
    (36, 37),
    (21, 38),
    (38, 39),
    (39, 40),
    (40, 41)
]  # 관절 접근 순서(트리구조)

HAND = {
    "WRIST": 0,
    "THUMB_1": 1,
    "THUMB_2": 2,
    "THUMB_3": 3,
    "THUMB_4": 4,
    "INDEX_FINGER_1": 5,
    "INDEX_FINGER_2": 6,
    "INDEX_FINGER_3": 7,
    "INDEX_FINGER_4": 8,
    "MIDDLE_FINGER_1": 9,
    "MIDDLE_FINGER_2": 10,
    "MIDDLE_FINGER_3": 11,
    "MIDDLE_FINGER_4": 12,
    "RING_FINGER_1": 13,
    "RING_FINGER_2": 14,
    "RING_FINGER_3": 15,
    "RING_FINGER_4": 16,
    "PINKY_1": 17,
    "PINKY_2": 18,
    "PINKY_3": 19,
    "PINKY_4": 20,

    "WRIST2": 21,
    "THUMB2_1": 22,
    "THUMB2_2": 23,
    "THUMB2_3": 24,
    "THUMB2_4": 25,
    "INDEX_FINGER2_1": 26,
    "INDEX_FINGER2_2": 27,
    "INDEX_FINGER2_3": 28,
    "INDEX_FINGER2_4": 29,
    "MIDDLE_FINGER2_1": 30,
    "MIDDLE_FINGER2_2": 31,
    "MIDDLE_FINGER2_3": 32,
    "MIDDLE_FINGER2_4": 33,
    "RING_FINGER2_1": 34,
    "RING_FINGER2_2": 35,
    "RING_FINGER2_3": 36,
    "RING_FINGER2_4": 37,
    "PINKY2_1": 38,
    "PINKY2_2": 39,
    "PINKY2_3": 40,
    "PINKY2_4": 41
}

#ratio_list = [0.6558401698431905, 0.6031772336473383, 0.6414785096985494, 0.5417740657623864, 0.6172197899783678, 0.5580510574162947, 0.6484368446692033, 0.7232914087536735, 0.6716514538215645, 0.5945155288376719, 0.9312276815456029, 0.956058386884934, 0.7340312564191465, 0.6450123954925382, 1.0359759333770044, 0.9866868794054321, 0.7762965246211136, 0.7785968209470062, 1.0703503922034507, 0.9865471779080366, 0.6702414688310544, 0.5949342377895633, 0.6739585710030499, 0.6443849500785759, 0.6309579854025523, 0.6365582726333331, 0.7473869071029396, 0.8926113278578914, 0.6652924229043587, 0.7377393560593201, 0.9181059600647433, 0.8881190967144201, 0.6893923972166318, 0.6869818926157439, 0.9535981594745762, 0.9389562544010447, 0.699263096676369, 0.6323390146681539, 0.8994543737508193, 0.9989295243362932]
#diff = [0.5556877255439758, 0.11814308166503906, -3.658419416296965e-07]

ratio_list = [0.8611112247507237, 0.8512248045497322, 0.7429211574686708, 0.654744372205626, 0.7965146029025739, 0.7321952835870895, 0.7567124376470022, 0.7280236109960389, 0.7731530606226698, 0.7485476077170597, 0.792894145539213, 0.7124843316218841, 0.7758625953503663, 0.767322148360031, 0.7571664420289933, 0.742970955450262, 0.779438972617196, 0.7424966778818443, 0.8043458808840726, 0.7960900426783876, 0.8192006147052723, 0.8006308291898538, 0.7060453005802338, 0.7822916636603819, 0.7865344026791256, 0.7796680106384162, 0.7797425770176459, 0.8148614591516958, 0.7717791762187562, 0.8333316815280362, 0.8104276072080211, 0.8289535957268765, 0.7568128182828981, 0.8293457354597432, 0.7641179219182914, 0.7714256032964484, 0.7532032726076066, 0.7941987346434214, 0.7890492466078824, 0.791117577347208]
diff = [0.4823681116104126, 0.09919661283493042, -1.3844947943653096e-07]

# 탐색 순서 트리 구조
def build_tree(order):
    tree = {}
    # 부모-자식 관계를 딕셔너리에 저장
    for parent, child in order:
        if parent not in tree:
            tree[parent] = []
        tree[parent].append(child)

    return tree

def normalization_setting(first_landmarks):
    # 정규화 하기 전에 실행 -> 각 관절별로 비율을 저장
    global ratios, root, tree, diffrence
    ratios = []

    user_example = first_landmarks
    std_example = [
        [0.26661598682403564, 0.711597740650177, 4.874195838056039e-07],
        [0.32163935899734497, 0.7184560894966125, -0.029448874294757843],
        [0.37858232855796814, 0.6977903246879578, -0.04312770441174507],
        [0.42299121618270874, 0.6848411560058594, -0.054915353655815125],
        [0.4590681493282318, 0.6842761635780334, -0.06737786531448364],
        [0.36708471179008484, 0.5721474885940552, -0.032141342759132385],
        [0.4064783751964569, 0.5135841369628906, -0.051575206220149994],
        [0.4313860535621643, 0.4770185947418213, -0.06675972789525986],
        [0.45146670937538147, 0.44173458218574524, -0.07870135456323624],
        [0.33268579840660095, 0.5425925850868225, -0.033596351742744446],
        [0.35936880111694336, 0.46274828910827637, -0.05313730612397194],
        [0.37758344411849976, 0.41014721989631653, -0.07062303274869919],
        [0.3907519280910492, 0.36271607875823975, -0.08317196369171143],
        [0.2954343855381012, 0.5364490747451782, -0.0379662849009037],
        [0.3048211336135864, 0.455700159072876, -0.05951618775725365],
        [0.31231608986854553, 0.40437719225883484, -0.07671330869197845],
        [0.3202109634876251, 0.3575412631034851, -0.08792506158351898],
        [0.25776025652885437, 0.5493021011352539, -0.04472137242555618],
        [0.25314879417419434, 0.4874737858772278, -0.06652303040027618],
        [0.2506535053253174, 0.4464271068572998, -0.07836546748876572],
        [0.25277867913246155, 0.406032919883728, -0.08543018996715546],
        [0.7200760841369629, 0.7297391295433044, 4.769220254274842e-07],
        [0.662412703037262, 0.7355905771255493, -0.02788771688938141],
        [0.6091150045394897, 0.7197923064231873, -0.0475865975022316],
        [0.5669642686843872, 0.708770215511322, -0.06445261090993881],
        [0.529534637928009, 0.7073748111724854, -0.08239232003688812],
        [0.6197328567504883, 0.5900130271911621, -0.038518331944942474],
        [0.5809513330459595, 0.531770646572113, -0.05946280434727669],
        [0.5570370554924011, 0.49436962604522705, -0.07640492171049118],
        [0.5368029475212097, 0.4582115411758423, -0.08977183699607849],
        [0.6565333604812622, 0.5608547925949097, -0.042210422456264496],
        [0.6338381171226501, 0.4793328046798706, -0.06011245399713516],
        [0.6214398145675659, 0.42643076181411743, -0.07630382478237152],
        [0.6113208532333374, 0.3782091736793518, -0.0886886864900589],
        [0.6962199807167053, 0.5581510066986084, -0.04842507466673851],
        [0.6875941157341003, 0.47771546244621277, -0.0696619525551796],
        [0.6816174387931824, 0.4279787242412567, -0.08520790189504623],
        [0.6746912598609924, 0.38150763511657715, -0.09612446278333664],
        [0.7360718250274658, 0.574323832988739, -0.05635146051645279],
        [0.7441684007644653, 0.5150861740112305, -0.07762540131807327],
        [0.7483400702476501, 0.47501587867736816, -0.08648458868265152],
        [0.7498372197151184, 0.4346314072608948, -0.0919022411108017]
    ]

    # 평행이동을 위함(기준점) - (x, y, z)

    diffrence = (user_example[0][0] - std_example[0][0], user_example[0][1] - std_example[0][1], user_example[0][2] - std_example[0][2])
    print("차이",diffrence[0], diffrence[1], diffrence[2])
    user_landmark = []
    std_landmark = []

    # 튜플로 제공된 데이터 리스트로 가공
    for i in range(42):
        user_landmark.append(
            [user_example[i][0], user_example[i][1], user_example[i][2]]
        )
        std_landmark.append(
            [std_example[i][0] + diffrence[0], std_example[i][1] + diffrence[1], std_example[i][2] + diffrence[2]]
        )

    def calculate_distance(point1, point2):
        """두 점 사이의 거리를 계산합니다."""
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) # 유클리드 거리 공식

    def calculate_ratio(root, tree, user_landmark, std_landmark): # 비율 리스트 생성
        calculate_result = [] # 비율 리스트

        # 재귀적으로 전위탐색 수행
        def traverse(node):
            nonlocal calculate_result
            if node in tree:
                for child in tree[node]:
                    user_kp1 = user_landmark[index[node]]
                    user_kp2 = user_landmark[index[child]]
                    std_kp1 = std_landmark[index[node]]
                    std_kp2 = std_landmark[index[child]]
                    result = calculate_distance(std_kp1, std_kp2) / calculate_distance(
                        user_kp1, user_kp2
                    )
                    calculate_result.append(result)
                    traverse(child)

        traverse(root)
        return calculate_result

    root = 0
    tree = build_tree(order)
    ratios1 = calculate_ratio(root, tree, user_landmark, std_landmark) # 비율 리스트
    print("[ratios1]",ratios1)

    root = 21
    ratios2 = calculate_ratio(root, tree, user_landmark, std_landmark)
    print("[ratios2]",ratios2)

    ratios = ratios1 + ratios2
    print(ratios)

    return ratios


def normalization(normalize_data):
    # 정규화: 선수 골격을 사용자에 맞춤
    std_landmarks = []  # 전체 사이클
    for landmarks in normalize_data:
        std_landmark = []  # 한 사이클

        # 튜플로 제공된 데이터 리스트로 가공
        for i in range(42):
            std_landmark.append(
                [landmarks[i][0] + diff[0], landmarks[i][1] + diff[1], landmarks[i][2] + diff[2]]
                #[landmarks[i][0], landmarks[i][1], landmarks[i][2]]
            )

        def normalize_keypoints(std_kp1, std_kp2, ratio):
            if ratio > 1:
                Px = (std_kp2[0] + ratio * std_kp1[0]) / (1 + ratio)
                Py = (std_kp2[1] + ratio * std_kp1[1]) / (1 + ratio)
                Pz = (std_kp2[2] + ratio * std_kp1[2]) / (1 + ratio)
            elif ratio < 1:
                Px = (std_kp2[0] - (1 - ratio) * std_kp1[0]) / (1 - (1 - ratio))
                Py = (std_kp2[1] - (1 - ratio) * std_kp1[1]) / (1 - (1 - ratio))
                Pz = (std_kp2[2] - (1 - ratio) * std_kp1[2]) / (1 - (1 - ratio))
            else:
                Px, Py, Pz = std_kp2
            return [Px, Py, Pz]

        # 정규화
        def normalize(root, tree, std_landmark):
            # 재귀적으로 전위탐색 수행
            def traverse(node):
                if node in tree:
                    for child in tree[node]:
                        std_kp1 = std_landmark[index[node]]
                        std_kp2 = std_landmark[index[child]]
                        if index[child] < 22:
                            ratio = ratio_list[index[child]-1]
                        else:
                            ratio = ratio_list[index[child]-2]
                        # ratio = ratio_list[index[node]]
                        # print(ratio)
                        result = normalize_keypoints(std_kp1, std_kp2, ratio)
                        std_landmark[index[child]] = result
                        parallel_move(child, tree, std_landmark, std_kp2, result)
                        traverse(child)

            traverse(root)

        def parallel_move(root, tree, std_landmark, std_kp2, result):
            diff = (result[0] - std_kp2[0], result[1] - std_kp2[1], result[2] - std_kp2[2])

            # 재귀적으로 전위탐색 수행
            def traverse(node):
                if node in tree:
                    for child in tree[node]:
                        Px, Py, Pz = std_landmark[index[child]]
                        temp_x = Px + diff[0]
                        temp_y = Py + diff[1]
                        temp_z = Pz + diff[2]
                        std_landmark[index[child]] = [temp_x, temp_y, temp_z]
                        traverse(child)

            traverse(root)
        root = 0
        tree = build_tree(order)
        normalize(root, tree, std_landmark)

        root = 21
        normalize(root, tree, std_landmark)

        std_landmarks.append(std_landmark)
    print(std_landmark)

    return std_landmarks