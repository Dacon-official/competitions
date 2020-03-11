1. 사전 데이터 수집 방법 (크롤링, 외부데이터 정제) 
(external_dataset -> preprocessing_code 폴더
pickle종합(train, test에서 나오는 위도 경도 기반으로 새로운 행정동 생성) -> 
second_whole_dict.pickle 생성 완료 (원본은 preprocessing_code.zip파일) (종합은 dacon에서 편집)

제주공항 강수량 전처리 https://amo.kma.go.kr/new/html/stat/stat1.jsp
(hourly 제주공항 9월 강수량, 10월 강수량 -> 0시부터 11시까지의 실제 측정 데이터만 활용) -> hourly_rain.csv
(일별 자료 8,9,10월 -> daily_rain.csv 생성) -> 이후 전날 강수량 사용
+ 전날 강수량으로 변수 넣어두고, 실제 측정 데이터도 merge -> 이상 없음

제주공항 전운량 전처리 https://amo.kma.go.kr/new/html/stat/stat1.jsp
hourly_cloud.csv (9,10월 11시까지의 실제 측정 데이터)

제주도_실거주자수 https://www.jeju.go.kr/open/stats/list/population_temp.htm?act=view&seq=1165007
2018년도 실거주자수 통계 -> 제주도_거주자수.csv

제주공항 도착 고객 
https://www.airport.co.kr/www/extra/dailyExpect/dailyExpectList/layOut.do?cid=2016053109481920258&menuId=4757 (일일예상승객정보) -> df_airport.csv -> 이후 진행되는 모든 _port 추가가 된 ipynb 파일의 추가된 부분은 모두 이 데이터로 인한 차이임





2. 데이터 수집 이후 전처리 (Create_Features 폴더
FE.ipynb -> 정상 실행 확인
FE_port.ipynb -> FE에서 df_airport를 이용한 것만 추가됨 -> 정상 실행 확인

3. catboost, lightgbm (Modeling 폴더)
Catoobst -> 0.25비율 (len_seeds = 5) (10시간 넘게 걸림) ->
cat_5_seeds_stractified5k_bus_route_id.csv

Lgbm-port -> 0.5비율 (len_seeds = 5) (2시간 걸림) ->
lgbm_5_seeds_stractified5k_bus_route_id_port.csv

Lgbm -> 0.25비율 (len_seeds = 40) (3시간 걸림) -> lgbm_40_seeds_stractified5k_bus_route_id.csv

4. submission 폴더 (앙상블 비율 설정)
final_submission.ipynb -> cat5seeds_lgbm40seeds_lgbmport5seeds_ensemble.csv 생성