from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
import time 
import os 
import urllib.request 
from multiprocessing import Pool 
import pandas as pd

# keyword=['중학생 증명사진',''중학생]
# keyword=['김유정 고등학생', '고등학생 증명사진', '채영 졸업사진', '이채연 졸업사진', '전소미 졸업사진', '한림예고 졸업사진', '서공예 졸업사진', '장원영 졸업사진']
# keyword=['쯔위', '아린', '이재욱', '유연정', '전소미', '최유정', '김유정', '위키미키 도연', '우주소녀 여름', '우주소녀 다영', '엘리스 벨라', '세븐틴 디노', '아스트로 라키', '빅톤 정수빈']
# keyword=['김세정', '이하이', '김영대', '이찬원', '박보검', '우도환', '제니', '한지현', '이현우', '엑소 디오', '엑소 세훈', '김호중', '고아성', '뷔', '임영웅']
# keyword=['효린', '소녀시대 수영', '소녀시대 서현', '박보영', '이엘리야', '초아', '박서준', '정해인', '유이', '유빈', '김수현', '폴킴', '유병재', '임시완', '옥택연', '에릭남', '배우 이지훈', '고경표', '한효주', '문근영']
# keyword=['류준열', '변요한', '유노윤호', '이장우', '박유천', '유아인', '민효린', '가수 나비', '유인영', '신민아', '모델 한혜진', '손담비', '테이', '박해진', '영탁', '이특', '장희진']
# keyword=['조인성', '배우 정우', '진태현', '개코', '김승현', '김래원', '김지석', '배우 김지훈', '봉태규', '조여정', '송지효', '박예진', '이진혁', '전지현', '배우 한혜진', '한예슬', '장동민', '유민상', '홍진경', '데프콘', '장민호', 전현무]
# keyword=['백지영', '배우 유선', '배우 김선영', '박선영', '안정환', '홍경민', '조진웅', '차태현', '이선균', '김준호', '배우 김정은', '문소리', '배해선', '엄태웅', '김제동', '이종혁', '송은이', '김서형', '신은경', '홍지민', '유재석', '윤정수']
# keyword=['김정난', '정웅인', '이서진', '신동엽', '마동석', '정준하', '유희열', '배우 이정은', '박미선', '배우 김혜선', '윤종신', '주영훈', '정준호', '지석진', '김광규', '이재용', '김건모', '이문식']
# keyword=['김국진', '팽현숙', '박해미', '이원종', '한석규', '김미화', '안내상', '배종옥', '최민수', '최수종', '최민식', '김일우', '정수라']
keyword=['이순재', '금보라', '나문희', '전원주', '최불암', '고두심', '강부자', '이덕화', '가수 현미', '하춘화', '김수미', '이경규']


def createFolder(directory): # os를 임포트 하면 폴더 만들 수 있음
    try: 
        if not os.path.exists(directory): #고구마 라는 폴더가 없으면 만들고, 있으면 넘어감
            os.makedirs(directory) 
    except OSError: 
        print ('Error: Creating directory. ' + directory)

def image_download(keyword): 
    createFolder('./'+keyword)
    chromedriver = 'c:/Program Files/Google/Chrome/chromedriver' 
    driver = webdriver.Chrome(chromedriver) 
    driver.implicitly_wait(time_to_wait=3)  # 3초동안 페이지가 로딩되는 걸 기다려준다는 의미
    '''
    implicitly wait : 웹페이지 전체가 넘어올때까지 기다리기
    explicitly wait : 웹페이지의 일부분이 나타날때까지 기다리기
    '''


################################### 구글 이미지 검색 접속 및 검색어 입력 ###################################
    print(keyword, '검색')  # 키워드 검색하도록 명령
    driver.get('https://www.google.co.kr/imghp?hl=ko')  # 크롤링할 URL입력
    Keyword=driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input') # xpath의 원하는 값을 가져옴
    
    '''
    find_element_by_xpath 
    /: 절대 경로를 나타냄
    //: 문서내에서 검색
    //@href : href속성이 있는 모든 태그 선택
    //a[@href='http://google.com']: a 태그의 href 속성에 http://google.com 속상값을 가진 모든 태그 선택
    (//a)[3]: 문서의 세 번째 링크 선택
    (//table)[last()]: 문서의 마지막 테이블 선택
    (//a)[position()< 3]: 문서의 처음 두링크선택
    //table/tr/* : 모든 테이블에서 모든 자식 tr 태그 선택
    //div[@*] : 속성이 하나라도 있는 div 태그 선택

    '''

    Keyword.send_keys(keyword)  # 키워드 입력
    driver.find_element_by_xpath('//*[@id="sbtc"]/button').click()  # 검색 버튼 누르기


################################################# 스크롤 #################################################
    print(keyword+' 스크롤 중 .............') 
    elem = driver.find_element_by_tag_name("body") 
    for i in range(1): 
        elem.send_keys(Keys.PAGE_DOWN)  # 무한 스크롤
        time.sleep(0.1) 
    try: 
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[4]/div[2]/input').click()  # 결과 더보기 버튼 누르기
        for i in range(1): 
            elem.send_keys(Keys.PAGE_DOWN) 
            time.sleep(0.1) 
    except: 
        pass 


############################################### 이미지 개수 ###############################################
    images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd") 
    print(keyword+' 찾은 이미지 개수:', len(images))
    links=[] 
    for i in range(1,len(images)): 
        try: 
            driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').click() 
            # links.append(driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute('src')) 
            links.append(driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img').get_attribute('src')) # 검사 copy x_path
            driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[2]/a').click() 
            print(keyword+' 링크 수집 중..... number :'+str(i)+'/'+str(len(images))) 
        except: 
            continue 

############################################# 이미지 다운로드 #############################################
        forbidden=0 
        for k,i in enumerate(links): 
            try: 
                url = i 
                start = time.time() 
                urllib.request.urlretrieve(url, "./"+keyword+"/"+keyword+"_"+str(k-forbidden)+".jpg")
                print(str(k+1)+'/'+str(len(links))+' '+keyword+' 다운로드 중....... Download time : '+str(time.time() - start)[:5]+' 초') 
            except: 
                forbidden+=1 
                continue 

    print(keyword+' ---다운로드 완료---')
    driver.close()

if __name__=='__main__': 
    pool = Pool(processes=4) # 4개의 프로세스를 사용합니다.     # Pool : 처리할 일들을 바닥에 뿌려놓고 알아서 분산처리 
    pool.map(image_download, keyword)

print('끝')
