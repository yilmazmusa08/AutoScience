import pandas as pd
import numpy as np
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
import warnings
from fitter import Fitter
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore')


#CLASS MODEL
class KolonTipiTahmini1:
    def __init__(self, threshold=20):
        self.threshold=threshold
        
    def fit_transform(self, data, columns=None):
        if not isinstance(data, pd.DataFrame):
            data=pd.DataFrame(data)
        if len(data) > 5000:
            data=data.sample(n=5000)
            
        distm=data.apply(lambda x: np.mean(pd.Series(x).value_counts(normalize=True).values))#değişkenin benzersiz değerlerinin frekans dağılımının normalleştirilmiş değerlerinin ortalamasını hesaplar.
        dists=data.apply(lambda x: np.std(pd.Series(x).value_counts(normalize=True).values))# benzersiz değerlerinin frekans dağılımının normalleştirilmiş değerlerinin standart sapmasını hesaplar.
        wc=data.apply(lambda x: round(len(str(x).split()) / len(str(x)), 5))#her bir değişkenin ortalama kelime sayısını hesaplar.
        wcs=data.apply(lambda x: np.std([len(str(kelime).split()) for kelime in x]))#her bir değişkenin kelime sayılarının standart sapmasını hesaplar.
        Len=data.apply(lambda x: len(str(x)) / len(x))#her bir değişkenin ortalamada kaç karakter içerdiğini hesaplar.
        lens=data.apply(lambda x: np.std([len(str(s).replace(" ", "")) for s in x]))# her bir değişkenin karakter sayılarının standart sapmasını hesaplar.
        uniq=data.apply(lambda x: pd.Series(x).nunique())#her bir değişkenin benzersiz değerlerinin sayısını hesaplar.
        most=data.apply(lambda x: pd.Series(x).value_counts().idxmax() if len(x)>0 and not x.isnull().all() else np.nan)#her bir değişkendeki en sık görülen değeri hesaplar.
        mostR=data.apply(lambda x: pd.Series(x).value_counts(normalize=True).values[0] if len(x)>0 and not x.isnull().all() else np.nan)#her bir değişkendeki en sık görülen değerin oranını hesaplar.
        uniqR=data.apply(lambda x: pd.Series(x).nunique() / len(x))# her bir değişkendeki benzersiz değerlerin oranını hesaplar.
        allunique=data.apply(lambda x: int(x.nunique() == len(x)))#her bir değişkendeki değerlerin hepsinin benzersiz olup olmadığını hesaplar.
        kolon_role=[]
        
        if columns:
            data=data[columns]

        for col in data.columns:
            if len(data[col].unique()) == 1:
                data.drop(col, axis=1, inplace=True)
            elif (len(data[col].unique()) == 2):
                kolon_role.append('flag')
            elif len(data[col].unique()) == len(data):
                kolon_role.append('unique')
            elif (all(isinstance(val, int) for val in data[col]) and len(data[col].unique()) == len(data) and set(data[col]) == set(range(1, len(data)+1))):
                kolon_role.append('id')
            elif (all(isinstance(val, int) for val in data[col]) and (re.search(r'(id|ID|Id|iD>|ıd)', col))):
                kolon_role.append('id')
            elif all(isinstance(val, int) for val in data[col]) and len(data[col].unique()) == len(data):
                digits=len(str(data[col].iloc[0]))
                if (data[col].apply(lambda x: len(str(x)) == digits).sum() / len(data[col])) > 0.9:
                    kolon_role.append('id')           
            elif all([isinstance(val, str) and pd.to_datetime(val, errors='coerce') is not pd.NaT for val in data[col].values]) or (data[col].dtype == 'datetime64[ns]'):
                    kolon_role.append('date')
            elif isinstance(data[col].iloc[0], str) and data[col].str.split().str.len().mean() > 4:
                kolon_role.append('text')
            elif len(data[col].unique()) > self.threshold and (all(isinstance(val, int) for val in data[col]) or all(isinstance(val, float) for val in data[col])):
                if data[col].apply(lambda x: (isinstance(x, (int,float)) and len(str(x).split('.')) > 1 and str(x).split('.')[1] != '0' and str(x).split('.')[1] != '0000')).sum() > 0:
                    kolon_role.append('scalar')
                elif data[col].apply(lambda x: isinstance(x, (int))).all():
                    kolon_role.append('numeric')
                else:
                    kolon_role.append('numeric')
            elif isinstance(data[col].iloc[0], str) and len(data[col].unique()) >= 0.9*len(data) and data[col].str.split().str.len().mean()<5:
                kolon_role.append('indentifier')     
            elif len(data[col].unique()) > self.threshold and (all(isinstance(val, int) for val in data[col]) or all(isinstance(val, float) for val in data[col])):
                kolon_role.append('numeric')
            elif len(data[col].unique()) < self.threshold:
                kolon_role.append('categoric')
            else:
                kolon_role.append("indentifier")

        if 'date' in kolon_role:
            date_cols=[col for col, role in zip(data.columns, kolon_role) if role == 'date']
            for date_col in date_cols:
                col_idx=data.columns.get_loc(date_col)
                kolon_role[col_idx]='date'
        #Eğer kolon değerleri datetime ise o zaman hangı formatda olduğunu gosterir
        def get_Date_Format(data) -> str:
            if not isinstance(data, pd.DataFrame):
                dataframe=pd.DataFrame(data)

            date_formats=[   
                "%m/%Y",
                "%m-%Y",    
                "%d/%m/%y",  
                "%m/%d/%y",  
                "%d.%m.%Y",  
                "%d/%m/%Y", 
                "%m/%d/%Y",  
                "%Y-%m-%d",  
                "%Y/%m/%d",  
                "%m-%d-%Y",  
                "%d-%m-%Y", 
                "%d.%m.%y",
                "%m.%d.%y",
                "%Y/%m",
                "%Y-%m",
                "%m/%d",
                "%d.%m",
                "%d/%m",
                "%m.%d",
                "%Y",
                "%d-%m"
                ]
            dict1={}
            for column in data.columns:
                values=data[column]
                for f in date_formats:
                    try:
                        date=pd.to_datetime(values, format=f)
                        if date.dt.strftime(f).eq(values).all():
                            dict1=f
                            break
                    except ValueError:
                        pass
                else: 
                    dict1=None
            return dict1
        date_formats = data.apply(lambda col: get_Date_Format(col.to_frame()))

        sonuc=[]
        for i in data.columns:
            distm_val=round(distm[i], 5) if distm.get(i) is not None else None
            dists_val=round(dists[i], 5) if dists.get(i) is not None else None
            wc_val=round(wc[i], 5) if wc.get(i) is not None else None
            wcs_val=round(wcs[i], 5) if wcs.get(i) is not None else None
            len_val=round(Len[i], 5) if Len.get(i) is not None else None
            lens_val=round(lens[i], 5) if lens.get(i) is not None else None
            uniq_val=round(uniq[i], 5) if uniq.get(i) is not None else None
            most_val=most[i] if most.get(i) is not None else None
            mostR_val=round(mostR[i], 5) if mostR.get(i) is not None else None
            uniqR_val=round(uniqR[i], 5) if uniqR.get(i) is not None else None
            allunique_val=round(allunique[i], 5) if allunique.get(i) is not None else None
            kolon_role_val=kolon_role[data.columns.get_loc(i)]
            top3=data[i].value_counts().nlargest(3).index.tolist()
            sonuc.append({
                'col_name': i,
                'distm': distm_val,
                'date_formats': date_formats[i],
                'dists': dists_val,
                "wc":wc_val,
                "wcs":wcs_val,
                "len":len_val,
                "lens":lens_val,
                "uniq":uniq_val,
                "uniqR %":uniqR_val,
                "most":most_val,
                "mostR %":mostR_val,
                "allunique":allunique_val,
                "top3":top3,
                'Role': kolon_role_val})

        result={}
        for d in sonuc:
            col_name=d.pop('col_name')
            result[col_name]=d
        return result


def analysis(df: pd.DataFrame, target=None, columns=None, warning=True, threshold_col=0.1, threshold_target=0.5, return_stats: bool = False):
    """----Bu fonksiyon veri setini analiz etmek için tasarlanmıştır.Fonksiyonun aldığı parametreler şunlardır:
    Parametreler:
    _____________
    1-df: analiz edilecek veri seti
    2-threshold_col: sütunlar arasındaki yüksek korelasyon için eşik değeri (varsayılan olarak 0.8)
    3-threshold_target: hedef değişkenle sütunlar arasındaki yüksek korelasyon için eşik değeri (varsayılan olarak 0.4)
    4-target: hedef değişkenin belirten bir boolean (varsayılan olarak None)
    5-columns: analiz edilecek sütunların listesi (varsayılan olarak tüm sütunlar seçilir)
    6-warning:warning parametresi, herhangi bir uyarı mesajı görüntülemek için kullanılabilir. 
    Bu uyarı mesajı, kullanıcının daha sonra yapacağı işlemleri yönlendirmesine yardımcı olabilir.Warning = True
    olduğu zaman warning'i olan sütünları siler.
    7-return_stats: veri seti istatistiklerini içeren bir dict döndürmek için bir boolean (varsayılan olarak False)"""

    high_corr_cols = []
    high_corr_target = []
    kt = KolonTipiTahmini1()

    if columns:  # Model (Class)--> kuruluyor
        data_dict = {}
        for col in columns:
            data_dict[col] = kt.fit_transform(df[[col]])[col]
    else:
        data_dict = kt.fit_transform(df)

# returun=True,returun=False
# ==========================

    # target object değilse kolerasyon hesaplanacak
    if target is not None and df[target].dtype == 'object':
        corr = False
    else:
        corr = True

    if corr:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix=df[numeric_columns].corr().abs()
        high_corr_target = []
        for col in corr_matrix.columns:
            if target is not None and col != target and corr_matrix[target][col] >= threshold_target:
                high_corr_target.append(col)

# Bu kodda bir veri seti için farklı olasılık dağılım fonksiyonlarının uygunluğu test edilir.
# ===========================================================================================
    for col in df:
        try:  # Her sütun için bir hata ayıklama bloğu başlatılır.
            # Sütunun tam sayı türünde olup olmadığını kontrol etmek için tüm satırların tamsayı olup olmadığını kontrol eden bir lambda işlevi tanımlanır ve apply() yöntemi kullanarak tüm sütuna uygulanır. Sonuçlar all() yöntemi ile kontrol edilir ve is_int değişkenine atanır.
            is_int = df[col].apply(lambda x: x.is_integer()).all()
            if is_int:
                # Eğer tüm satırlar tamsayı ise, sütunun veri tipi 'int' olarak değiştirilir.
                df[col] = df[col].astype("int")
        except:
            pass  # Hata oluşursa geçilir.
    result_dict = {}
    distributions = ['norm', 'uniform', 'binom', 'poisson', 'gamma', 'beta', 'lognorm', 'weibull_min', 'weibull_max', 'expon', 'pareto', 'cauchy', 'chi', 'f', 't', 'laplace',
                     'bernoulli', 'exponential', 'geometric', 'hypergeometric', 'normal_mix', 'rayleigh', 'student_t', 'weibull']  # Kullanılacak dağılımların listesi oluşturulur.
    # float64' veri tipine sahip tüm sütunlar için bir döngü
    for col in df.select_dtypes(include=['float64']).columns:
        if len(df) >= 5000:
            # örnekleme işlemi verinin daha hızlı işlenmesine yardımcı olmak için
            df_sampled = df.sample(n=5000, random_state=42)
        else:
            df_sampled = df  # satır sayısı 2000'den küçükse orijinal veri kullanılır
        # kullanılacak dağılımlar parametre olarak verilir.
        f = Fitter(df_sampled[col], distributions=distributions)
        f.fit()
        # Uygunluk hesaplamalarında hata oluştuysa devam edilir.
        if len(f.df_errors) > 0:
            try:
                # En uygun dağılımı belirleyen get_best() yöntemi kullanılır ve sonuçlar sözlük olarak döndürülür. En uygun dağılımın adı alınır ve best_dist değişkenine atanır.
                best_dist = list(f.get_best().keys())[0]
            except KeyError:  # Eğer en uygun dağılım adı elde edilemezse geçilir.
                continue

            # dağılım adı 'weibull_min' veya 'weibull_max' ise 'weibull' olarak değiştirilir.
            if best_dist == 'weibull_min' or best_dist == 'weibull_max':
                best_dist = 'weibull'
            # dağılım adı 'lognormvariate' ise 'lognorm' olarak değiştirilir.
            elif best_dist == 'lognormvariate':
                best_dist = 'lognorm'
            # dağılım adı 'norm' ise 'gauss' olarak değiştirilir.
            elif best_dist == 'norm':
                best_dist = 'gauss'
            # En uygun dağılım adı, sütun adı ile birlikte sözlüğe eklenir.
            result_dict[col] = best_dist

# Problem tipinin ve metrik belirlenmesi
# ======================================
    if target:
        corr_dict = {}  # Korelasyon dict oluşturuluyor
        nan_ratio_list = []  # NaN oranları listesi oluşturuluyor
        warning_list = []  # warning listesi oluşturuluyor
        corr_deneme = []  # kolonlar arası dusuk korelasyonu bulmak için
    
        # Eğer veri setindeki sütun sayısı 5'ten büyükse ve hedef değişkenin veri tipi int64 veya float64 ise ve benzersiz değer sayısı 20'den büyükse problem_type değişkenine 'scoring' atanır
        if len(df.columns) > 5 and df[target].dtype in ['int64', 'float64'] and df[target].nunique() > 20:
            problem_type = 'scoring'
            metrics = ['r2_score', 'mean_absolute_error', 'mean_squared_error']
        elif df[target].nunique() == 2:  # Eğer hedef değişkenin benzersiz değer sayısı 2 ise
            min_count = df[target].value_counts().min()
            # hedef değişkenin sınıflarının sayısı %1'lik dilimdeki örnek sayısından daha küçükse:
            if min_count < 0.01 * len(df):
                # problem_type değişkenine 'anomaly detection' atanır.
                problem_type = 'anomaly detection'
                metrics = ['precision_score', 'recall_score', 'f1_score']
            else:  # aksi takdirde:
                # problem_type değişkenine 'binary classification' atanır.
                problem_type = 'binary classification'
                metrics = ['f1_score', 'precision_score', 'recall_score']
        # Eğer hedef değişkenin benzersiz değer sayısı 2 ile 20 arasındaysa:
        elif 2 < df[target].nunique() < 20:
            # problem_type değişkenine 'multi-class classification' atanır.
            problem_type = 'multi-class classification'
            metrics = ['accuracy_score', 'precision_score', 'recall_score']
        # Eğer veri setindeki sütun sayısı 5'ten küçükse:
        elif len(df.columns) < 5:
            for col in df:
                # veri setindeki her sütunun veri tipi 'object' ise:
                if df[col].dtype == 'object':
                    try:
                        # sütunların 'datetime64[ns]' veri tipine dönüştürülebilir ise:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            if any(df[col].dtype == 'datetime64[ns]' for col in df.columns):
                # problem_type değişkenine 'time series' atanır.
                problem_type = 'time series'
                metrics = ['mae', 'mse', 'rmse']
            # veri setindeki sütun sayısı 5'ten küçük ve integer veri tipine sahip sadece 2 sütun varsa veya herhangi bir sütunun adında 'id', 'ID', 'Id', 'iD>', 'ıd' kelimesi geçiyorsa:
            elif len(df.columns) < 5 and len([col for col in df.columns if isinstance(df[col], "int")]) == 2 or any(re.search(r'(id|ID|Id|iD>|ıd)', col) for col in df.columns):
                # problem_type değişkenine 'recommendation' atanır.
                problem_type = 'recommendation'
                metrics = ['recall_score', 'precision_score', 'map_score']
        # problem_type sözlük olarak {'problem_type': problem_type, 'metrics': metrics} şeklinde atanır.
        problem_type = {'problem_type': problem_type, 'metrics': metrics}

# target aktif olduğu zaman -- Bu kod bloğu, NaN, sparse ve unique değerleri kontrol eder ve sayısal sütunlar arasındaki yüksek korelasyonu kontrol eder.Eğer warning TRue olursa warning kolonları siler
# =======================================================================================================================================================================================================

        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        print(numeric_columns)

        korelasyonlar = {}
        if isinstance(target, int):
            corr_matrix = df[numeric_columns].corr()[[target]]
            print('corr_matrix', corr_matrix)
            corr_matrix = corr_matrix.drop(target)
            kolonlar = list(corr_matrix.index)
            for i in range(len(kolonlar)):
                korelasyonlar[kolonlar[i]] = corr_matrix.iloc[i, 0]
        else:
            pass




        for col in df.columns: # Veri setindeki her kolon için:
            if df[col].dtype == "int64" or df[col].dtype == "float64":# Eğer kolon sayısal ise:
                null_values = df[col].isna().sum() # NaN değerlerinin sayısı hesaplanıyor
                zero_values = (df[col] == 0).sum()# Sıfır değerlerinin sayısı hesaplanıyor
                total_values = len(df[col])  # Toplam satır sayısı hesaplanıyor
                null_ratio = null_values / total_values  # NaN oranı hesaplanıyor
                zero_ratio = zero_values / total_values  # Sıfır oranı hesaplanıyor
                unique_ratio = len(df[col].unique()) / total_values# Benzersiz değer oranı hesaplanıyor
                ratio_str = ""  # Oranların string olarak birleştirileceği değişken tanımlanıyor

                if null_ratio > 0:  # Eğer NaN oranı sıfırdan büyükse:
                    ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)# NaN oranı stringe ekleniyor
                if zero_ratio > 0 and null_ratio == 0: # Eğer sıfır oranı sıfırdan büyük ve NaN oranı sıfırsa:
                    if len(ratio_str) > 0:  # Daha önce oran eklendi ise:
                        ratio_str += ", "  # Stringe virgül ekleniyor
                    ratio_str += "sparse:{:.2f}%".format(zero_ratio * 100) # Sıfır oranı stringe ekleniyor
                if unique_ratio >= 0.95 and null_ratio == 0 and zero_ratio == 0: # Eğer benzersiz değer oranı %95'ten büyük, NaN ve sıfır oranları sıfırsa:
                    if len(ratio_str) > 0:  # Daha önce oran eklendi ise:
                        ratio_str += ", "  # Stringe virgül ekleniyor
                    ratio_str += "unique:{:.2f}%".format(unique_ratio * 100) # Benzersiz değer oranı stringe ekleniyor
                if null_ratio > 0:  # eğer sütunda NaN oranı varsa:
                    ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)  # oranlar stringine NaN oranını ekler.
                if unique_ratio >= 0.95 and null_ratio == 0:# eğer sütunda benzersiz değerlerin oranı %95'ten fazlaysa ve NaN oranı yoksa:
                    if len(ratio_str) > 0:  # eğer oranlar stringi doluysa:
                        ratio_str += ", "  # oranlar stringine virgül ekler.
                    ratio_str += "unique:{:.2f}%".format(unique_ratio * 100)# oranlar stringine benzersiz değer oranını ekler.
                    if warning:  # warning değişkeni True ise:
                        if problem_type.get("problem_type") != 'anomaly detection': # problem_type  anomaly detection ise wagningl'ler hesplanacak warning_liste' eklenmeyecek
                            warning_list.append(col)  # warning listesine sütunu ekler.
                if null_ratio > 0.5:  # eğer sütundaki NaN oranı %50'den fazlaysa:
                    if warning:  # warning değişkeni True ise:
                        if problem_type.get("problem_type") != 'anomaly detection':
                            warning_list.append(col)# warning listesine sütunu ekler.
                if problem_type.get("problem_type") != 'anomaly detection':
                    if zero_ratio > 0.8:
                        warning_list.append(col)

                
                corr = df.select_dtypes(include=np.number).corr()[col]# Korelasyon matrisindeki korelasyon değerlerini hesapla
                high_corr_cols = corr[(corr > threshold_col) & (corr.index != col) & (
                    corr.index.isin(df.select_dtypes(include=np.number).columns))].index.tolist()# Yüksek korelasyonlu sütunların isimleri
                if high_corr_cols:# Eğer yüksek korelasyonlu sütun varsa, corr_dict'e ekle
                    corr_dict[col] = {"high_corr_with": high_corr_cols}
                    corr_deneme.append(col)
                
                if ratio_str or high_corr_cols:# Eğer NaN oranı veya yüksek korelasyonlu sütun varsa, nan_ratio_list'e ekle
                    nan_ratio = {"column": col}
                    if ratio_str:
                        nan_ratio["ratio"] = ratio_str
                    if col in corr_dict:
                        nan_ratio.update(corr_dict[col])
                    nan_ratio_list.append(nan_ratio)

            
            else:# Eğer sütun numerik değil ise:
                
                null_values = df[col].isna().sum()# NaN değerlerin sayısını hesapla
                total_values = len(df[col]) # Toplam değer sayısını hesapla
                null_ratio = null_values / total_values# NaN oranını hesapla
                zero_values = (df[col] == 0).sum()
                zero_ratio = zero_values / total_values
                unique_ratio = len(df[col].unique()) / total_values# Benzersiz değer oranını hesapla
                ratio_str = ""  # boş bir oranlar stringi oluşturur.

                if null_ratio > 0:  # eğer sütunda NaN oranı varsa:
                    ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)# oranlar stringine NaN oranını ekler.
                if unique_ratio >= 0.95 and null_ratio == 0:# eğer sütunda benzersiz değerlerin oranı %95'ten fazlaysa ve NaN oranı yoksa:
                    if len(ratio_str) > 0:  # eğer oranlar stringi doluysa:
                        ratio_str += ", "  # oranlar stringine virgül ekler.
                    ratio_str += "unique:{:.2f}%".format(unique_ratio * 100)# oranlar stringine benzersiz değer oranını ekler.
                    warning_list.append(col)  # warning listesine sütunu ekler.
                if null_ratio > 0.5:  # eğer sütundaki NaN oranı %50'den fazlaysa:
                    warning_list.append(col)  # warning listesine sütunu ekler.

                if ratio_str:  # eğer oranlar stringi doluysa:
                    nan_ratio = {"column": col}# sütun adı ile bir sözlük oluşturur.
                    nan_ratio["ratio"] = ratio_str  # warning listesine sütunu ekler.
                    nan_ratio_list.append(nan_ratio)# NaN oranları listesine sözlüğü ekler.

               
        """sütunlar arasındaki korelasyonları hesaplar.
            Daha sonra, hedef değişken ile diğer sütunlar arasındaki korelasyonu ölçmek 
                için hedef değişken ile  sütunlar arasındaki korelasyonları alır. Daha sonra, 
                    hedef değişken ile korelasyonu en düşük olan sütunu seçer ve warning_list listesine ekler.
                        Bu şekilde, target ile düşük korelasyona sahip sütunlar modelleme sürecinde göz ardı edilebilir."""

        if problem_type.get("problem_type") != 'anomaly detection':
            if len(corr_deneme) > 0:
                test = {key: korelasyonlar[key] for key in corr_deneme if key in korelasyonlar}
                if test:
                    min_value = min(abs(val) for val in test.values())
                    min_key = [key for key, value in test.items() if abs(value) == min_value][0]
                    if warning:
                        warning_list.append(min_key)


# target aktif olduğu zaman-veri setindeki hedef değişkeni kullanarak özellik seçimi (feature_importance) yapar.
# =============================================================================================================
        # Silme işlemi
        null_counts = df.isnull().sum()
        empty_cols = null_counts[null_counts >= len(df) * 0.6].index
        df.drop(empty_cols, axis=1, inplace=True)

        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 20:
                df[col].fillna(df[col].mode()[0], inplace=True)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            elif df[col].dtype != 'object':
                df[col].fillna(df[col].mean(), inplace=True)



        X = df.select_dtypes(include=['float', 'int'])
        if target in X.columns:
            X.drop(target, axis=1, inplace=True)

        if target is not None:
            if len(X) > 5000:
                X = X.sample(n=5000, random_state=42)
                y = df.loc[X.index, target]
            else:
                y = df[target]

        feature_names = X.columns
        rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, oob_score=True, max_depth=5)

        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
        feat_selector.fit(X.values, y.values)
        importance = feat_selector.ranking_

        feature_names = X.columns
        feature_importance = {}
        total_importance = 0
        for i in range(len(feature_names)):
            feature_importance[feature_names[i]] = importance[i]
            total_importance += importance[i]

        for feature in feature_importance:
            feature_importance[feature] = round((feature_importance[feature] / total_importance) * 100, 2)

# Warning hesaplaması target aktif olmadığı zaman ve warning None olduğu zaman. Tek değişiklik(if warning is None:)
# ==================================================================================================================
    if target is None:
        clustering = 'clustering'
        clustering = {clustering}
        corr_dict = {}
        nan_ratio_list = []
        warning_list = []
        problem_type = {}
        for col in df.columns:  # Veri setindeki her kolon için:
            if df[col].dtype == "int64" or df[col].dtype == "float64":# Eğer kolon sayısal ise:
                null_values = df[col].isna().sum() # NaN değerlerinin sayısı hesaplanıyor
                zero_values = (df[col] == 0).sum() # Sıfır değerlerinin sayısı hesaplanıyor
                total_values = len(df[col])  # Toplam satır sayısı hesaplanıyor
                null_ratio = null_values / total_values  # NaN oranı hesaplanıyor
                zero_ratio = zero_values / total_values  # Sıfır oranı hesaplanıyor
                unique_ratio = len(df[col].unique()) / total_values # Benzersiz değer oranı hesaplanıyor
                ratio_str = ""  # Oranların string olarak birleştirileceği değişken tanımlanıyor
                if null_ratio > 0:  # Eğer NaN oranı sıfırdan büyükse:
                    ratio_str += "NaN:{:.2f}%".format(null_ratio * 100) # NaN oranı stringe ekleniyor
                if zero_ratio > 0 and null_ratio == 0:# Eğer sıfır oranı sıfırdan büyük ve NaN oranı sıfırsa:
                    if len(ratio_str) > 0:  # Daha önce oran eklendi ise:
                        ratio_str += ", "  # Stringe virgül ekleniyor
                    ratio_str += "sparse:{:.2f}%".format(zero_ratio * 100)# Sıfır oranı stringe ekleniyor
                if unique_ratio >= 0.95 and null_ratio == 0 and zero_ratio == 0:# Eğer benzersiz değer oranı %95'ten büyük, NaN ve sıfır oranları sıfırsa:
                    if len(ratio_str) > 0:  # Daha önce oran eklendi ise:
                        ratio_str += ", "  # Stringe virgül ekleniyor       
                    ratio_str += "unique:{:.2f}%".format(unique_ratio * 100) # Benzersiz değer oranı stringe ekleniyor

                if null_ratio > 0:
                    ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)
                if unique_ratio >= 0.95 and null_ratio == 0:
                    if len(ratio_str) > 0:
                        ratio_str += ", "
                    ratio_str += "unique:{:.2f}%".format(unique_ratio * 100)
                    warning_list.append(col)
                if null_ratio > 0.5:
                    warning_list.append(col)

                if zero_ratio > 0.8:
                    warning_list.append(col)

               
                corr = df.select_dtypes(include=np.number).corr()[col] # Korelasyon matrisindeki korelasyon değerlerini hesapla
                high_corr_cols = corr[(corr > threshold_col) & (corr.index != col) & (
                    corr.index.isin(df.select_dtypes(include=np.number).columns))].index.tolist() # Yüksek korelasyonlu sütunların isimleri
                if high_corr_cols:# Eğer yüksek korelasyonlu sütun varsa, corr_dict'e ekle
                    corr_dict[col] = {"high_corr_with": high_corr_cols}

               
                if ratio_str or high_corr_cols: # Eğer NaN oranı veya yüksek korelasyonlu sütun varsa, nan_ratio_list'e ekle
                    nan_ratio = {"column": col}
                    if ratio_str:
                        nan_ratio["ratio"] = ratio_str
                    if col in corr_dict:
                        nan_ratio.update(corr_dict[col])
                    nan_ratio_list.append(nan_ratio)

           
            else: # Eğer sütun numerik değil ise:
              
                null_values = df[col].isna().sum() # NaN değerlerin sayısını hesapla
                total_values = len(df[col]) # Toplam değer sayısını hesapla
                null_ratio = null_values / total_values# NaN oranını hesapla
                zero_values = (df[col] == 0).sum()
                zero_ratio = zero_values / total_values
                unique_ratio = len(df[col].unique()) / total_values# Benzersiz değer oranını hesapla
                ratio_str = ""  # boş bir oranlar stringi oluşturur.
                if null_ratio > 0:
                    ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)
                if unique_ratio >= 0.95 and null_ratio == 0:
                    if len(ratio_str) > 0:
                        ratio_str += ", "
                    ratio_str += "unique:{:.2f}%".format(unique_ratio * 100)
                    warning_list.append(col)
                if null_ratio > 0.5:
                    warning_list.append(col)

                if ratio_str:
                    nan_ratio = {"column": col}
                    nan_ratio["ratio"] = ratio_str
                    nan_ratio_list.append(nan_ratio)

# Warning=False olduğu zaman
# ==========================
    if warning is False:
        if target:
            corr_dict = {}
            nan_ratio_list = []
            warning_list = []
            corr_deneme = []  # kolonlar arasında dusuk corr bulmak için

            numeric_columns = df.select_dtypes(include='number').columns
            corr_matrix = df[numeric_columns].corr()[[target]]
            print('cor_matrix', corr_matrix)

            corr_matrix = corr_matrix.drop(target)
            kolonlar = list(corr_matrix.index)
            korelasyonlar = {}

            for i in range(len(kolonlar)):
                korelasyonlar[kolonlar[i]] = corr_matrix.iloc[i, 0]


            for col in df.columns:  # Veri setindeki her kolon için:
                if df[col].dtype == "int64" or df[col].dtype == "float64": # Eğer kolon sayısal ise:
                    null_values = df[col].isna().sum() # NaN değerlerinin sayısı hesaplanıyor
                    zero_values = (df[col] == 0).sum()# Sıfır değerlerinin sayısı hesaplanıyor
                    total_values = len(df[col]) # Toplam satır sayısı hesaplanıyor
                    null_ratio = null_values / total_values  # NaN oranı hesaplanıyor
                    zero_ratio = zero_values / total_values  # Sıfır oranı hesaplanıyor
                    unique_ratio = len(df[col].unique()) / total_values # Benzersiz değer oranı hesaplanıyor
                    ratio_str = ""  # Oranların string olarak birleştirileceği değişken tanımlanıyor

                    if null_ratio > 0:  # Eğer NaN oranı sıfırdan büyükse:
                        ratio_str += "NaN:{:.2f}%".format(null_ratio * 100) # NaN oranı stringe ekleniyor
                    if zero_ratio > 0 and null_ratio == 0:# Eğer sıfır oranı sıfırdan büyük ve NaN oranı sıfırsa:
                        if len(ratio_str) > 0:  # Daha önce oran eklendi ise:
                            ratio_str += ", "  # Stringe virgül ekleniyor
                        ratio_str += "sparse:{:.2f}%".format(zero_ratio * 100)# Sıfır oranı stringe ekleniyor
                    if unique_ratio >= 0.95 and null_ratio == 0 and zero_ratio == 0: # Eğer benzersiz değer oranı %95'ten büyük, NaN ve sıfır oranları sıfırsa:
                        if len(ratio_str) > 0:  # Daha önce oran eklendi ise:
                            ratio_str += ", "  # Stringe virgül ekleniyor
                        ratio_str += "unique:{:.2f}%".format(unique_ratio * 100)# Benzersiz değer oranı stringe ekleniyor

                    if null_ratio > 0:
                        ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)
                    if unique_ratio >= 0.95 and null_ratio == 0:
                        if len(ratio_str) > 0:
                            ratio_str += ", "
                        ratio_str += "unique:{:.2f}%".format(unique_ratio * 100)# oranlar stringine benzersiz değer oranını ekler.
                        if warning is False:
                            if problem_type.get("problem_type") != 'anomaly detection':
                                warning_list.append(col)  # warning listesine sütunu ekler.
                    if null_ratio > 0.5:  # eğer sütundaki NaN oranı %50'den fazlaysa:
                        if warning is False:
                            if problem_type.get("problem_type") != 'anomaly detection':#problem tipi anomaly detection değilse
                                warning_list.append(col)# warning listesine sütunu ekler.
                    if problem_type.get("problem_type") != 'anomaly detection':
                        if zero_ratio > 0.8:
                            warning_list.append(col)

                    
                    corr = df.select_dtypes(include=np.number).corr()[col]# Korelasyon matrisindeki korelasyon değerlerini hesapla              
                    high_corr_cols = corr[(corr > threshold_col) & (corr.index != col) & (
                        corr.index.isin(df.select_dtypes(include=np.number).columns))].index.tolist()# Yüksek korelasyonlu sütunların isimleri
                   
                    if high_corr_cols: # Eğer yüksek korelasyonlu sütun varsa, corr_dict'e ekle
                        corr_dict[col] = {"high_corr_with": high_corr_cols}
                        corr_deneme.append(col)

                    
                    if ratio_str or high_corr_cols:# Eğer NaN oranı veya yüksek korelasyonlu sütun varsa, nan_ratio_list'e ekle
                        nan_ratio = {"column": col}
                        if ratio_str:
                            nan_ratio["ratio"] = ratio_str
                        if col in corr_dict:
                            nan_ratio.update(corr_dict[col])
                        nan_ratio_list.append(nan_ratio)

               
                else: # Eğer sütun numerik değil ise:
                  
                    null_values = df[col].isna().sum() # NaN değerlerin sayısını hesapla
                    total_values = len(df[col])# Toplam değer sayısını hesapla
                    null_ratio = null_values / total_values # NaN oranını hesapla
                    zero_values = (df[col] == 0).sum()
                    zero_ratio = zero_values / total_values
                    unique_ratio = len(df[col].unique()) / total_values # Benzersiz değer oranını hesapla
                    ratio_str = ""  # boş bir oranlar stringi oluşturur.

                    if null_ratio > 0:
                        ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)
                    if unique_ratio >= 0.95 and null_ratio == 0:
                        if len(ratio_str) > 0:
                            ratio_str += ", "
                        ratio_str += "unique:{:.2f}%".format(
                            unique_ratio * 100)
                        warning_list.append(col)
                    if null_ratio > 0.5:
                        warning_list.append(col)

                    if ratio_str:
                        nan_ratio = {"column": col}
                        nan_ratio["ratio"] = ratio_str
                        nan_ratio_list.append(nan_ratio)

            if problem_type.get("problem_type") != 'anomaly detection':
                if len(corr_deneme) > 0:
                    test = {key: korelasyonlar[key]
                            for key in corr_deneme if key in korelasyonlar}
                    min_value = min(abs(val) for val in test.values())
                    min_key = [key for key, value in test.items() if abs(
                        value) == min_value][0]
                    if warning is False:
                        warning_list.append(min_key)
                        
                        
# return_stats = True olduğunda
# ============================
    if return_stats:
        if target:
            if warning is False:
                result = {
                    "Role": data_dict,
                    "Warning List": warning_list,
                    "Warning": nan_ratio_list,
                    "distributions": result_dict,
                    "high_corr_target": high_corr_target,
                    "feature_importance": {k: f"{v}%" for k, v in feature_importance.items()},
                    "problem_type": problem_type
                }
                return result

            if warning:
                missing_columns = [col for col in warning_list if col in df.columns]
                df = df.drop(missing_columns, axis=1)
                df_head = df.head(1)
                df_head = df_head.to_json(orient="records")
                result = {
                    "Role": data_dict,
                    "Warning List": warning_list,
                    "Warning": nan_ratio_list,
                    "distributions": result_dict,
                    "feature_importance": {k: f"{v}%" for k, v in feature_importance.items()},
                    "problem_type": problem_type,
                    "df_head": df_head}

                return result

        if target is None:
            if warning is False:
                result = {
                    "Role": data_dict,
                    "Warning List": warning_list,
                    "Warning": nan_ratio_list,
                    "distributions": result_dict,
                    "problem_type": clustering}
                return result
        
            if warning is True:
                missing_columns = [col for col in warning_list if col in df.columns]
                df = df.drop(missing_columns, axis=1)
                df_head = df.head(1)
                df_head = df_head.to_json(orient="records")
                result = {
                    "Role": data_dict,
                    "Warning List": warning_list,
                    "Warning": nan_ratio_list,
                    "distributions": result_dict,
                    "problem_type": clustering}
                return result
            


# return_stats = False olduğu zaman
# ===============================
    else:
        if warning is False:
            if target is None:
                result = {
                    "Role": {k: v['Role'] for k, v in data_dict.items()},
                    "Warning List": warning_list,
                    "Warning": nan_ratio_list,
                    "distributions": result_dict,
                    "problem_type": clustering
                }

                return result

        else:
            if target is None:
                df = df.drop(warning_list, axis=1)
                df_head = df.head(1)
                df_head = df_head.to_json(orient="records")
                result = {
                    "Role": {k: v['Role'] for k, v in data_dict.items()},
                    "Warning List": warning_list,
                    "Warning": nan_ratio_list,
                    "distributions": result_dict,
                    "problem_type": clustering,
                    "df_head": df_head}

                return result

        if warning:
            if target:
                df = df.drop(warning_list, axis=1)
                
                df_head = df.head(1)
                df_head = df_head.to_json(orient="records")
                result = {
                    "Role": {k: v['Role'] for k, v in data_dict.items()},
                    "Warning List": warning_list,
                    "Warning": nan_ratio_list,
                    "distributions": result_dict,
                    "high_corr_target": high_corr_target,
                    "feature_importance": {k: f"{v}%" for k, v in feature_importance.items()},
                    "problem_type": problem_type,
                    "df_head": df_head
                }
                return result

            else:
                result = {
                    "Role": {k: v['Role'] for k, v in data_dict.items()},
                    "Warning List": warning_list,
                    "Warning": nan_ratio_list,
                    "distributions": result_dict,
                    "high_corr_target": high_corr_target,
                    "feature_importance": {k: f"{v}%" for k, v in feature_importance.items()},
                    "problem_type": problem_type
                }
                return result


# Principal Component
def calculate_pca(df, comp_ratio=1.0, target=None):
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() < 20:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
    if target is not None:
        df = df.drop(columns=[target])
    df.fillna(df.mean(), inplace=True)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    pca = PCA()
    pca.fit(df[numeric_cols])
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(explained_var_ratio)
    n_components = np.searchsorted(cumsum_var_ratio, comp_ratio) + 1
    pca = PCA(n_components=n_components)
    pca.fit(df[numeric_cols])
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(explained_var_ratio)
    result_dict = {
        'Cumulative Explained Variance Ratio': cumsum_var_ratio.tolist()}
    result_dict['Principal Component'] = list(
        range(1, len(explained_var_ratio)+1))
    return result_dict


def set_to_list(data):
    if isinstance(data, set):
        return list(data)
    if isinstance(data, dict):
        return {set_to_list(key): set_to_list(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [set_to_list(item) for item in data]
    if isinstance(data, np.int64):
        return int(data)
    return data


# df = pd.read_csv("train.csv")
# print(analysis(df,target="Survived",return_stats=True))
