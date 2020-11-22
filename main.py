from flask import Flask, escape, request,render_template, redirect, url_for, jsonify, session,flash
from flask_mysqldb import MySQL
from pymysql.cursors import DictCursor
from werkzeug.utils import secure_filename
import pymysql, yaml, cgi, itertools, re, os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import operator

app = Flask(__name__)
app.secret_key = '1010'
# Configure db
db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
#app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)


@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")


@app.route('/signup', methods=['POST', 'GET'])
def signup1():
    if request.method == 'POST':
        # Fetch form data
        userDetails = request.form

        name = userDetails['name']
        if name == '':
            flash('Enter name')
        gender = userDetails['gender']
        if gender == '':
            flash('Select gender')
        email = userDetails['email']
        if email == '':
            flash('Enter email')
        regex = '^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
        if (re.search(regex, email)):
            print("valid mail id")
        else :
            flash("Enter Valid Email id")
            return redirect('/signup')
        age = userDetails['age']
        if age == '':
            flash('Enter age')
        qual = userDetails['qualification']
        if qual == '---Select---':
            flash('Enter qualification')
        prev = userDetails['prev_kno']
        if prev == '---Select---':
            flash('Enter previous knowledge')
        interest = userDetails['interest']
        if interest == '---Select---':
            flash('Enter area of interest')
        password = userDetails['password']
        flag = 0
        while True:
            if (len(password) < 8):
                flag = -1
                flash('Password must be greater than 8 characters')
                break
            elif not re.search("[a-z]", password):
                flag = -1
                flash('Password must have atleast one small letter')
                break
            elif not re.search("[A-Z]", password):
                flag = -1
                flash('Password must have atleast one capital letter')
                break
            elif not re.search("[0-9]", password):
                flag = -1
                flash('Password must have atleast one alphabet')
                break
            elif not re.search("[_@$]", password):
                flag = -1
                flash('Password must have atleast one special symbol')
                break
            elif re.search("\s", password):
                flag = -1
                break
            else:
                flag = 0
                break

        if flag == -1:
            # flash("Not a Valid Password")
            return redirect('/signup')
        else:
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO register(name, email, age, gender, qual, pre_knowledge, area_interest, password) "
                        "VALUES(%s, %s, %s, %s, %s, %s, %s, %s)",
                        (name, email, age, gender, qual, prev, interest, password))
            mysql.connection.commit()
            if name != 0:
                return redirect('/login')
            else:
                flash('Wrong Credentials')
            cur.close()
    return render_template("signup.html", title='Sign up')

@app.route('/login', methods=['GET', 'POST'])
@app.route('/layoutInner')
def login1():
    error = None
    if request.method == 'POST':
        userDetails = request.form
        email = userDetails['email']
        password = userDetails['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM register WHERE email = %s AND password = %s ", (email, password))
        data1 = cur.fetchone()
        # print(data1)
        mysql.connection.commit()

        if data1:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['u_id'] = data1[0]
            session['name'] = data1[1]
            session['pre_knowledge'] = data1[6]
            session['area_interest'] = data1[7]
            session['qual'] = data1[4]
            session['age'] = data1[3]
            session['email'] = data1[2]
            session['gender'] = data1[5]
            return redirect('/dashboard')
        else:
            error = 'Wrong Credentials'
        cur.close()
    return render_template("login.html", title='Login', error=error)


@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('name', None)
    session.pop('u_id', None)
    return redirect('/')


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    cur = mysql.connection.cursor()
    cur.execute("SELECT course_name, status, rating FROM course_enrolled WHERE u_id = %s", [session['u_id']])
    courses = cur.fetchall()
    crs = [item for t in courses for item in t]
    b = crs
    if request.method == 'POST':
        # Fetch form data
        det = request.form
        val = det['edit1']
        if val != 0:
            return redirect('/edit')
    return render_template("profile.html", title="Profile", b=b, l=len(b))


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    error = None
    if request.method == 'POST':
        change = request.form
        name = change['name']
        gender = change['gender']
        # email = change['email']
        age = change['age']
        qual = change['qualification']
        prev = change['prev_kno']
        interest = change['interest']
        # password = change['password']
        cur = mysql.connection.cursor()

        if qual == '---select---':
            error = 'Enter qualification and age'
        else:
            cur.execute("UPDATE register SET name = %s, age = %s, qual = %s, gender = %s,"
                        " pre_knowledge = %s, area_interest = %s WHERE u_id = %s",
                        (name, age, qual, gender, prev, interest, session['u_id']))
            mysql.connection.commit()
            cur.close()
            session['pre_knowledge'] = prev
            session['area_interest'] = interest
            return redirect('/dashboard')
    return render_template("edit.html", title="edit_profile", error=error)


@app.route('/dashboard', methods = ['GET', 'POST'])
def course():
    if request.method == 'GET':
        pre = session['pre_knowledge']
        area = session['area_interest']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM courses WHERE T3 = %s", [area])
        avg= cur.fetchall()
        mysql.connection.commit()

        cur = mysql.connection.cursor()
        cur.execute("SELECT course_name FROM course_enrolled WHERE u_id = %s ORDER BY u_id asc", [session['u_id']])
        data51 = cur.fetchall()
        mysql.connection.commit()
        data515 = [item for t in data51 for item in t]
        # print(data515)
        cur = mysql.connection.cursor()
        data5 = None
        if data515 :
            cur.execute("SELECT course_name FROM courses WHERE course_name = %s AND T3 = %s",
                        (data515[-1], session['area_interest']))
            data5 = cur.fetchone()
            mysql.connection.commit()
            # print(data5)
        if data5:
            cur = mysql.connection.cursor()
            cur.execute("SELECT course_name FROM course_enrolled WHERE u_id = %s", [session['u_id']])
            data8 = cur.fetchall()
            data85 = [item for t in data8 for item in t]
            # print(data85)
            mysql.connection.commit()
            b = data5[0]

            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM courses")
            data3 = cur.fetchall()
            mysql.connection.commit()

            cur = mysql.connection.cursor()
            cur.execute("SELECT c_id, u_id, rating FROM course_enrolled")
            data7 = cur.fetchall()
            mysql.connection.commit()

            df1 = pd.DataFrame(data7, columns=['course_id', 'user_id', 'rating'])
            df2 = pd.DataFrame(avg, columns=['course_id', 'course_name', 'T1', 'T2', 'T3',
                                             'course_overview', 'enrolled','rating_count', 'avg_rating'])
            df = pd.merge(df1, df2, on='course_id')

            combine_course_rating = df.dropna(axis=0, subset=['course_name'])
            course_ratingCount = (combine_course_rating.
                groupby(by=['course_name'])['rating'].
                count().
                reset_index().
                rename(columns={'rating': 'totalRatingCount'})
            [['course_name', 'totalRatingCount']]
                )
            rating_with_totalRatingCount = combine_course_rating.merge(course_ratingCount, left_on='course_name',
                                                                       right_on='course_name', how='left')
            pd.set_option('display.float_format', lambda x: '%.3f' % x)
            popularity_threshold = 0
            rating_popular_course = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
            ## First lets create a Pivot matrix
            course_features_df = rating_popular_course.pivot_table(index='course_name', columns='user_id',
                                                                   values='rating').fillna(0)
            from scipy.sparse import csr_matrix
            course_features_df_matrix = csr_matrix(course_features_df.values)
            from sklearn.neighbors import NearestNeighbors
            model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
            model_knn.fit(course_features_df_matrix)

            def giv_rec_rate1(title):
                a = []
                query_index = course_features_df.index.get_loc(title)
                distances, indices = model_knn.kneighbors(course_features_df.iloc[query_index, :].values.reshape(1, -1),
                                                          n_neighbors=10)
                for i in range(0, len(distances.flatten())):
                    if i != 0:
                        a.append(course_features_df.index[indices.flatten()[i]])
                return (a)

            P = giv_rec_rate1(b)
            # print(P)
            for i in data85 :
                for j in P:
                    # print(i,j)
                    if (i == j):
                        P.remove(j)
                        # print(P1)
        else:
            # Calculating average weighted score
            df = pd.DataFrame(avg, columns=['c_id', 'course_name', 'T1', 'T2', 'T3', 'course_overview',
                                            'enrolled','rating_count', 'rating'])
            df = df.replace(r'\n', ' ', regex=True)
            # print(df['course_name'])
            df1 = df.copy()
            # Get names of indexes for which column Age has value 30
            indexNames = df1[df1['T1'] == pre].index

            # Delete these row indexes from dataFrame
            df1.drop(indexNames, inplace=True)
            # print(df1['course_name'])
            if len(df1) >= 4 :
                df2 = df1.copy()
                # print(1)
            else :
                df2=df.copy()
                # print(2)
            # print(df2)
            courses_column_renamed = df2.rename(index=str, columns={"enrolled": "popularity"})
            courses_cleaned_df = courses_column_renamed.drop(columns=['T1', 'T2', 'T3'])
            v = courses_cleaned_df['rating_count']
            R = courses_cleaned_df['rating']
            C = courses_cleaned_df['rating'].mean()
            m = courses_cleaned_df['rating_count'].quantile(0.70)
            courses_cleaned_df['weighted_average'] = ((R * v) + (C * m)) / (v + m)
            # print(courses_cleaned_df)

            course_sorted_ranking = courses_cleaned_df.sort_values('weighted_average', ascending=False)
            weight_average = course_sorted_ranking.sort_values('weighted_average', ascending=False)
            popularity = course_sorted_ranking.sort_values('popularity', ascending=False)

            from sklearn.preprocessing import MinMaxScaler
            scaling = MinMaxScaler()
            course_scaled_df = scaling.fit_transform(courses_cleaned_df[['weighted_average', 'popularity']])
            course_normalized_df = pd.DataFrame(course_scaled_df, columns=['weighted_average', 'popularity'])
            courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_normalized_df
            courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_scaled_df

            courses_cleaned_df['score'] = courses_cleaned_df['normalized_weight_average'] * 0.5 + courses_cleaned_df[
                'normalized_popularity'] * 0.5
            courses_scored_df = courses_cleaned_df.sort_values(['score'], ascending=False)

            scored_df = courses_cleaned_df.sort_values('score', ascending=False)
            weighted_score =scored_df.drop(columns=['c_id', 'course_overview','popularity','rating_count','rating',
                                                    'weighted_average','score','normalized_weight_average',
                                                    'normalized_popularity'])

            tup = [tuple(x) for x in weighted_score.values]
            P = [item for t in tup for item in t]
            # print(P)

        cur = mysql.connection.cursor()
        cur.execute("SELECT course_name FROM courses WHERE rating > 4")
        data1 = cur.fetchall()
        data1 = [item1 for t1 in data1 for item1 in t1]
        mysql.connection.commit()
        return render_template('dashboard.html', data=P, data1=data1)
        mysql.connection.commit()
    else:
        userDetails = request.form
        cou = userDetails['myCountry']
        if cou != 0:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM courses WHERE course_name = %s", [cou])
            data2 = cur.fetchone()
            session['c_id'] = data2[0]
            session['tag1'] = data2[2]
            # print(data2)
            mysql.connection.commit()
            if data2:
                # Create session data, we can access this data in other routes
                session['course_name'] = cou
                b = session['u_id']
                cur = mysql.connection.cursor()
                cur.execute("SELECT status FROM course_enrolled WHERE course_name = %s AND u_id = %s", (cou, b))
                c2 = cur.fetchone()
                if c2 == ('enrolled',):
                    return redirect('/enrolled_course')
                elif c2 == ('Completed',) :
                    return redirect('/complete_course')
                else:
                    return redirect('/courses')
            else:
                error = 'Wrong Credentials'
            cur.close()
    return render_template('dashboard.html')


@app.route('/courses', methods = ['GET', 'POST'])
def desc():
        if request.method == 'GET':
            course1 = session['course_name']
            cur = mysql.connection.cursor()
            cur.execute("SELECT status FROM course_enrolled WHERE course_name = %s AND u_id = %s",
                        (course1, session['u_id']))
            d = cur.fetchone()
            mysql.connection.commit()
            cur.close()
            # print("status",d)
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM courses where course_name = %s", [course1])
            data4 = cur.fetchall()
            data5 = [item for t in data4 for item in t]
            # print("if",data5)
            mysql.connection.commit()
            cur.close()
            if d == ('enrolled',) :
                return redirect('/enrolled_course')
            elif d == ('Completed',) :
                return redirect('/complete_course')
            else:
                # Recommendation Model
                if data5:
                    cur = mysql.connection.cursor()
                    cur.execute("SELECT course_name FROM course_enrolled WHERE u_id = %s", [session['u_id']])
                    data8 = cur.fetchall()
                    data85 = [item for t in data8 for item in t]
                    cur = mysql.connection.cursor()
                    cur.execute("SELECT * FROM courses")
                    data6 = cur.fetchall()
                    # data850 = [item for t in data3 for item in t]
                    # print(data850)
                    mysql.connection.commit()
                    df = pd.DataFrame(data6, columns=['course_id', 'course_name', 'T1', 'T2', 'T3',
                                                      'course_overview', 'enrolled','rating_count', 'avg_rating'])
                    df = df.replace(r'\n', '', regex=True)

                    courses_column_renamed = df.rename(index=str, columns={"enrolled": "popularity"})

                    courses_cleaned_df = courses_column_renamed.drop(columns=['T1', 'T2', 'T3'])

                    v = courses_cleaned_df['rating_count']
                    R = courses_cleaned_df['avg_rating']
                    C = courses_cleaned_df['avg_rating'].mean()
                    m = courses_cleaned_df['rating_count'].quantile(0.70)
                    courses_cleaned_df['weighted_average'] = ((R * v) + (C * m)) / (v + m)

                    course_sorted_ranking = courses_cleaned_df.sort_values('weighted_average', ascending=False)
                    weight_average = course_sorted_ranking.sort_values('weighted_average', ascending=False)
                    popularity = course_sorted_ranking.sort_values('popularity', ascending=False)

                    from sklearn.preprocessing import MinMaxScaler
                    scaling = MinMaxScaler()
                    course_scaled_df = scaling.fit_transform(courses_cleaned_df[['weighted_average', 'popularity']])
                    course_normalized_df = pd.DataFrame(course_scaled_df, columns=['weighted_average', 'popularity'])
                    courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_normalized_df
                    courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_scaled_df

                    # print(courses_cleaned_df[['normalized_weight_average','normalized_popularity']])

                    courses_cleaned_df['score'] = courses_cleaned_df['normalized_weight_average'] * 0.5 + \
                                                  courses_cleaned_df['normalized_popularity'] * 0.5
                    courses_scored_df = courses_cleaned_df.sort_values(['score'], ascending=False)

                    scored_df = courses_cleaned_df.sort_values('score', ascending=False)

                    # Content Based recommendation
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    # Using Abhishek Thakur's arguments for TF-IDF
                    tfv = TfidfVectorizer(min_df=3, max_features=None,
                                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                          ngram_range=(1, 3),
                                          stop_words='english')
                    # Filling NaNs with empty string
                    courses_cleaned_df['course_overview'] = courses_cleaned_df['course_overview'].fillna('')
                    # print(type(courses_cleaned_df))
                    # Fitting the TF-IDF on the 'overview' text
                    tfv_matrix = tfv.fit_transform(courses_cleaned_df['course_overview'])

                    from sklearn.metrics.pairwise import sigmoid_kernel
                    # Compute the sigmoid kernel
                    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

                    # Reverse mapping of indices and movie titles
                    indices = pd.Series(courses_cleaned_df.index, index=courses_cleaned_df['course_name'])\
                        .drop_duplicates()

                    def give_rec(title, sig=sig):
                        z = []
                        idx = indices[title]
                        # print(idx)
                        b = int(idx)
                        sig_scores = list(enumerate(sig[b]))
                        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
                        # print(sig[b])
                        sig_scores = sig_scores[1:10]
                        course_indices = [i[0] for i in sig_scores]
                        # print(course_indices)
                        for i in range(0, len(course_indices)):
                            z.append(courses_cleaned_df['course_name'].iloc[course_indices[i]])
                        return (z)

                    P = give_rec(course1)
                    # print(P)
                    for i in data85:
                        for j in P:
                            # print(i,j)
                            if (i == j):
                                P.remove(j)
                                # print(P1)
        else:
            userDetails = request.form
            enroll = userDetails['myCountry']
            if enroll == 'Enroll':
                cur = mysql.connection.cursor()
                cur.execute("SELECT * FROM courses WHERE course_name = %s", [session['course_name']])
                data10 = cur.fetchone()
                # print(data10[0])
                session['c_id'] = data10[0]
                en = 'enrolled'
                cur.execute("INSERT INTO course_enrolled(c_id, u_id, course_name, status)"
                            "VALUES(%s, %s, %s, %s)", (session['c_id'], session['u_id'], session['course_name'], en))
                mysql.connection.commit()
                cur.close()
                return redirect('/enrolled_course')
            else:
                session['course_name'] = enroll
                cur = mysql.connection.cursor()
                cur.execute("SELECT * FROM courses where course_name = %s", [session['course_name']])
                data4 = cur.fetchall()
                data5 = [item for t in data4 for item in t]
                mysql.connection.commit()
                cur.close()
                if data5:
                    # Create session data, we can access this data in other routes
                    #session['course_name'] = enroll
                    return redirect('/courses')
                else:
                    error = 'Wrong Credentials'
        return render_template("courses.html", title='Course content',data3 = data5, data4 = P)


@app.route('/enrolled_course', methods = ['GET', 'POST'])
def desc1():
    if request.method == 'GET':
        course1 = session['course_name']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM courses where course_name = %s", [course1])
        data4 = cur.fetchall()
        data5 = [item for t in data4 for item in t]
        # print(data3)
        mysql.connection.commit()
        cur.close()

        # Recommendation Model
        if data5:
            cur = mysql.connection.cursor()
            cur.execute("SELECT course_name FROM course_enrolled WHERE u_id = %s", [session['u_id']])
            data8 = cur.fetchall()
            data85 = [item for t in data8 for item in t]

            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM courses")
            data3 = cur.fetchall()
            # data850 = [item for t in data3 for item in t]
            # print(data850)
            mysql.connection.commit()
            df = pd.DataFrame(data3, columns=['course_id', 'course_name', 'T1', 'T2', 'T3', 'course_overview',
                                              'enrolled','rating_count', 'avg_rating'])
            df = df.replace(r'\n', '', regex=True)

            courses_column_renamed = df.rename(index=str, columns={"enrolled": "popularity"})

            courses_cleaned_df = courses_column_renamed.drop(columns=['T1', 'T2', 'T3'])

            v = courses_cleaned_df['rating_count']
            R = courses_cleaned_df['avg_rating']
            C = courses_cleaned_df['avg_rating'].mean()
            m = courses_cleaned_df['rating_count'].quantile(0.70)

            courses_cleaned_df['weighted_average'] = ((R * v) + (C * m)) / (v + m)

            course_sorted_ranking = courses_cleaned_df.sort_values('weighted_average', ascending=False)

            weight_average = course_sorted_ranking.sort_values('weighted_average', ascending=False)

            popularity = course_sorted_ranking.sort_values('popularity', ascending=False)

            from sklearn.preprocessing import MinMaxScaler
            scaling = MinMaxScaler()
            course_scaled_df = scaling.fit_transform(courses_cleaned_df[['weighted_average', 'popularity']])
            course_normalized_df = pd.DataFrame(course_scaled_df, columns=['weighted_average', 'popularity'])
            courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_normalized_df
            courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_scaled_df
            # print(courses_cleaned_df[['normalized_weight_average','normalized_popularity']])

            courses_cleaned_df['score'] = courses_cleaned_df['normalized_weight_average'] * 0.5 + \
                                          courses_cleaned_df['normalized_popularity'] * 0.5
            courses_scored_df = courses_cleaned_df.sort_values(['score'], ascending=False)

            scored_df = courses_cleaned_df.sort_values('score', ascending=False)

            # Content Based recommendation
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Using Abhishek Thakur's arguments for TF-IDF
            tfv = TfidfVectorizer(min_df=3, max_features=None,
                                  strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                  ngram_range=(1, 3),
                                  stop_words='english')
            # Filling NaNs with empty string
            courses_cleaned_df['course_overview'] = courses_cleaned_df['course_overview'].fillna('')
            # print(type(courses_cleaned_df))
            # Fitting the TF-IDF on the 'overview' text
            tfv_matrix = tfv.fit_transform(courses_cleaned_df['course_overview'])

            from sklearn.metrics.pairwise import sigmoid_kernel
            # Compute the sigmoid kernel
            sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

            # Reverse mapping of indices and movie titles
            indices = pd.Series(courses_cleaned_df.index, index=courses_cleaned_df['course_name']).drop_duplicates()

            def give_rec(title, sig=sig):
                z = []
                idx = indices[title]
                # print(idx)
                b = int(idx)
                sig_scores = list(enumerate(sig[b]))
                sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
                # print(sig[b])
                sig_scores = sig_scores[1:10]
                course_indices = [i[0] for i in sig_scores]
                # print(course_indices)
                for i in range(0, len(course_indices)):
                    z.append(courses_cleaned_df['course_name'].iloc[course_indices[i]])
                return (z)
            P = give_rec(course1)
            # print(P)
            for i in data85:
                for j in P:
                    # print(i,j)
                    if (i == j):
                        P.remove(j)
                        # print(P1)
    else:
        userDetails = request.form
        enroll = userDetails['myCountry']
        if enroll == 'Complete':
            rate = userDetails['rating']
            feed = userDetails['feedback']
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM courses WHERE course_name = %s", [session['course_name']])
            data10 = cur.fetchone()
            session['c_id'] = data10[0]
            en = 'Completed'
            cur.execute("UPDATE course_enrolled SET status = %s, rating = %s, feedback = %s WHERE u_id = %s AND"
                        " c_id = %s", (en, rate, feed, session['u_id'], session['c_id']))
            mysql.connection.commit()
            cur.close()
            cur = mysql.connection.cursor()
            cur.execute("UPDATE register SET pre_knowledge = %s WHERE u_id = %s", (session['tag1'],session['u_id']))
            mysql.connection.commit()
            cur.close()
            return redirect('/complete_course')
        else:
            # print("I am here")
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM courses WHERE course_name = %s", [enroll])
            data2 = cur.fetchone()
            # print(data2)
            mysql.connection.commit()
            cur.close()
            if data2:
                # Create session data, we can access this data in other routes
                session['course_name'] = enroll
                return redirect('/courses')
            else:
                error = 'Wrong Credentials'
            cur.close()
    # print(P)
    return render_template("courses_enr.html", title='Course content', data3=data5, data4=P)


@app.route('/complete_course', methods = ['GET', 'POST'])
def desc2():
    if request.method == 'GET':
        course1 = session['course_name']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM courses where course_name = %s", [course1])
        data1 = cur.fetchall()
        data = [item for t in data1 for item in t]
        mysql.connection.commit()
        cur.close()
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM course_enrolled WHERE course_name = %s AND u_id = %s",
                    (session['course_name'],session['u_id']))
        data3 = cur.fetchall()
        data2 = [item for t in data3 for item in t]
        mysql.connection.commit()
        cur.close()
        if data:
            cur = mysql.connection.cursor()
            cur.execute("SELECT course_name FROM course_enrolled WHERE u_id = %s", [session['u_id']])
            data8 = cur.fetchall()
            data85 = [item for t in data8 for item in t]

            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM courses")
            data3 = cur.fetchall()
            mysql.connection.commit()
            df = pd.DataFrame(data3, columns=['course_id', 'course_name', 'T1', 'T2', 'T3', 'course_overview',
                                            'enrolled', 'rating_count', 'avg_rating'])
            df = df.replace(r'\n', '', regex=True)

            courses_column_renamed = df.rename(index=str, columns={"enrolled": "popularity"})

            courses_cleaned_df = courses_column_renamed.drop(columns=['T1', 'T2', 'T3'])

            v = courses_cleaned_df['rating_count']
            R = courses_cleaned_df['avg_rating']
            C = courses_cleaned_df['avg_rating'].mean()
            m = courses_cleaned_df['rating_count'].quantile(0.70)

            courses_cleaned_df['weighted_average'] = ((R * v) + (C * m)) / (v + m)

            course_sorted_ranking = courses_cleaned_df.sort_values('weighted_average', ascending=False)

            weight_average = course_sorted_ranking.sort_values('weighted_average', ascending=False)

            popularity = course_sorted_ranking.sort_values('popularity', ascending=False)

            from sklearn.preprocessing import MinMaxScaler
            scaling = MinMaxScaler()
            course_scaled_df = scaling.fit_transform(courses_cleaned_df[['weighted_average', 'popularity']])
            course_normalized_df = pd.DataFrame(course_scaled_df, columns=['weighted_average', 'popularity'])
            courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_normalized_df
            courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_scaled_df

            # print(courses_cleaned_df[['normalized_weight_average','normalized_popularity']])

            courses_cleaned_df['score'] = courses_cleaned_df['normalized_weight_average'] * 0.5 + \
                                          courses_cleaned_df['normalized_popularity'] * 0.5
            courses_scored_df = courses_cleaned_df.sort_values(['score'], ascending=False)

            scored_df = courses_cleaned_df.sort_values('score', ascending=False)

            # Content Based recommendation
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Using Abhishek Thakur's arguments for TF-IDF
            tfv = TfidfVectorizer(min_df=3, max_features=None,
                                  strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                  ngram_range=(1, 3),
                                  stop_words='english')
            # Filling NaNs with empty string
            courses_cleaned_df['course_overview'] = courses_cleaned_df['course_overview'].fillna('')
            # print(type(courses_cleaned_df))
            # Fitting the TF-IDF on the 'overview' text
            tfv_matrix = tfv.fit_transform(courses_cleaned_df['course_overview'])

            from sklearn.metrics.pairwise import sigmoid_kernel
            # Compute the sigmoid kernel
            sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

            # Reverse mapping of indices and movie titles
            indices = pd.Series(courses_cleaned_df.index, index=courses_cleaned_df['course_name']).drop_duplicates()

            def give_rec(title, sig=sig):
                z = []
                idx = indices[title]
                # print(idx)
                b = int(idx)
                sig_scores = list(enumerate(sig[b]))
                sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
                # print(sig[b])
                sig_scores = sig_scores[1:10]
                course_indices = [i[0] for i in sig_scores]
                # print(course_indices)
                for i in range(0, len(course_indices)):
                    z.append(courses_cleaned_df['course_name'].iloc[course_indices[i]])
                return (z)

            P = give_rec(course1)
            for i in data85:
                for j in P:
                    # print(i,j)
                    if (i == j):
                        P.remove(j)
                        # print(P1)
    else:
        userDetails = request.form
        enroll = userDetails['myCountry']
        if enroll:
            session['course_name'] = enroll
            return redirect('/courses')
    return render_template("course_complete.html", title='Course content', data=data, data1=data2, data4=P)


@app.route('/trainer', methods=['GET', 'POST'])
def trainer():
    cur = mysql.connection.cursor()
    cur.execute("SELECT c_id FROM trainer_course WHERE t_id = %s", [session['t_id']])
    courses = cur.fetchall()
    crs = [item for t in courses for item in t]
    b = crs
    t = int(b[1])
    cur.close()
    cur = mysql.connection.cursor()
    cur.execute("SELECT courses.course_name FROM courses INNER JOIN trainer_course ON courses.c_id=trainer_course.c_id WHERE trainer_course.t_id=%s", [session['t_id']])
    c_name = cur.fetchall()
    crs = [i for t in c_name for i in t]
    val = crs
    cur.close()
    cur = mysql.connection.cursor()
    cur.execute("SELECT domain_kno FROM trainer WHERE t_id = %s", [session['t_id']] )
    data20 = cur.fetchone()
    # print(data20)
    cur.close()
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM courses WHERE T3 = %s",[data20])
    data201 = cur.fetchall()
    cur.close()
    cur = mysql.connection.cursor()
    cur.execute("SELECT course_overview FROM courses WHERE T3 = %s",[data20])
    data202 = cur.fetchall()
    data203 = [item for t in data202 for item in t]
    data204 = [element.lower() for element in data203]
    cur.close()
    # print(data202)
    # df = pd.read_csv(r"C:\Users\Arshad\PycharmProjects\flacrud\fla\Course_Info.csv")
    df = pd.DataFrame(data201, columns=['course_id', 'course_name', 'T1', 'T2', 'T3', 'course_overview', 'enrolled',
                                   'rating_count', 'avg_rating'])
    df = df.replace(r'\n', '', regex=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3),
                          stop_words='english')
    df['course_overview'] = df['course_overview'].fillna('')
    tfv_matrix = tfv.fit_transform(df['course_overview'])
    # print(tfv.get_feature_names())
    b1 = tfv.get_feature_names()
    # print(b1)
    a = [0 for i in range(len(b1))]
    b2 = ['python','tableau','css','html','android','android studio','javascript','php','sql','reactnative','java','xml','html5','machine learning','data science','web development']
    a = [0 for i in range(len(b2))]
    for i in data204:
        for x in range(len(b2)):
            a[x] = a[x] + i.count(b2[x])
    # print(a)
    count_words = {}
    for key in b2:
        for value in a:
            count_words[key] = value
            a.remove(value)
            break
    # print(count_words)
    sorted_d = dict(sorted(count_words.items(), key=operator.itemgetter(1), reverse=True))
    # print(sorted_d)
    # x1=[None]*2
    # for key, i in sorted_d.items(), range(2):
    #         x1[i] = key
    # print(x1)
    res = [0 for i in range(2)]
    res[0] = list(sorted_d.keys())[0]
    res[1] = list(sorted_d.keys())[1]
    # print(res)

    # print(data201)
    df1 = pd.DataFrame(data201, columns=['course_id', 'course_name', 'T1', 'T2', 'T3', 'course_overview', 'enrolled',
                                        'rating_count', 'rating'])
    df1 = df1.replace(r'\n', '', regex=True)
    courses_column_renamed = df1.rename(index=str, columns={"enrolled": "popularity"})
    courses_cleaned_df = courses_column_renamed.drop(columns=['T1', 'T2', 'T3'])
    v = courses_cleaned_df['rating_count']
    R = courses_cleaned_df['rating']
    C = courses_cleaned_df['rating'].mean()
    m = courses_cleaned_df['rating_count'].quantile(0.70)
    courses_cleaned_df['weighted_average'] = ((R * v) + (C * m)) / (v + m)
    # print(courses_cleaned_df)

    course_sorted_ranking = courses_cleaned_df.sort_values('weighted_average', ascending=False)
    weight_average = course_sorted_ranking.sort_values('weighted_average', ascending=False)
    popularity = course_sorted_ranking.sort_values('popularity', ascending=False)

    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler()
    course_scaled_df = scaling.fit_transform(courses_cleaned_df[['weighted_average', 'popularity']])
    course_normalized_df = pd.DataFrame(course_scaled_df, columns=['weighted_average', 'popularity'])
    courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_normalized_df
    courses_cleaned_df[['normalized_weight_average', 'normalized_popularity']] = course_scaled_df

    courses_cleaned_df['score'] = courses_cleaned_df['normalized_weight_average'] * 0.5 + courses_cleaned_df[
        'normalized_popularity'] * 0.5
    courses_scored_df = courses_cleaned_df.sort_values(['score'], ascending=False)

    scored_df = courses_cleaned_df.sort_values('score', ascending=False)
    weighted_score = scored_df.drop(columns=['course_id', 'course_overview', 'popularity', 'rating_count', 'rating',
                                             'weighted_average', 'score', 'normalized_weight_average',
                                             'normalized_popularity'])

    tup = [tuple(x) for x in weighted_score.values]
    P = [item for t in tup for item in t]
    # print(P)
    if request.method == 'POST':
        # Fetch form data
        det = request.form
        val1 = det['tc1']
        print(val1)
        # val2 = det['myCountry']
        if val1 != 0:
            session['course_name'] = val1
            return redirect('/trainer_course')
        elif val2 != 0:
            session['course_name'] = val2
            return redirect('/courses')
    return render_template("trainer.html", title = 'Trainer dashboard',  b=b, l=len(b), val = val, res=res, data = P)


@app.route('/trainer_login', methods = ['GET', 'POST'])
def trainerlogin():
    error = None
    if request.method == 'POST':
        userDetails = request.form
        email = userDetails['email']
        password = userDetails['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM trainer WHERE t_email = %s AND password = %s ", (email, password))
        tdata = cur.fetchone()
        # print(data1)
        mysql.connection.commit()
        cur.close()
        if tdata:
            # Create session data, we can access this data in other routes
            session['logged_in'] = True
            session['t_id'] = tdata[0]
            session['name'] = tdata[1]
            session['gender'] = tdata[2]
            session['email'] = tdata[3]
            session['qual'] = tdata[4]
            session['index'] = tdata[5]
            session['techex'] = tdata[6]
            session['domkno'] = tdata[7]
            return redirect('/trainer')
        else:
            error = 'Wrong Credentials'
    return render_template("trainerLogin.html", title = 'Trainer login')


@app.route('/trainer_signup', methods = ['GET', 'POST'])
def trainersignup():
    if request.method == 'POST':
        # Fetch form data
        userDetails = request.form

        name = userDetails['name']
        if name == '':
            flash('Enter name')
        gender = userDetails['gender']
        if gender == '':
            flash('Select gender')
        email = userDetails['email']
        if email == '':
            flash('Enter email')
        # age = userDetails['age']
        # if age == '':
        #     flash('Enter age')
        qual = userDetails['qualification']
        if qual == '---Select---':
            flash('Enter qualification')
        index = userDetails['ind_ex']
        # if index == '---Select---':
        #     flash('Enter previous knowledge')
        techex = userDetails['tech_ex']
        # if techex == '---Select---':
        #     flash('Enter area of interest')
        dom = userDetails['dom']
        if dom == '---Select---':
            flash('Enter domain knowledge')
        password = userDetails['password']
        flag = 0
        while True:
            if (len(password) < 8):
                flag = -1
                flash('Password must be greater than 8 characters')
                break
            elif not re.search("[a-z]", password):
                flag = -1
                flash('Password must have atleast one small letter')
                break
            elif not re.search("[A-Z]", password):
                flag = -1
                flash('Password must have atleast one capital letter')
                break
            elif not re.search("[0-9]", password):
                flag = -1
                flash('Password must have atleast one alphabet')
                break
            elif not re.search("[_@$]", password):
                flag = -1
                flash('Password must have atleast one special symbol')
                break
            elif re.search("\s", password):
                flag = -1
                break
            else:
                flag = 0
                break
        if flag == -1:
            # flash("Not a Valid Password")
            return redirect('/trainer_signup')
        else:
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO trainer(t_name, t_gender, t_email, t_qualification, industry_ex, teaching_ex,"
                " domain_kno, password) VALUES(%s, %s, %s, %s, %s, %s, %s, %s)",
                (name, gender, email, qual, index, techex, dom, password))
            mysql.connection.commit()
            if name != 0:
                return redirect('/trainer_login')
            else:
                flash('Wrong Credentials')
            cur.close()
    return render_template("trainerSignup.html", title = 'Trainer Signup')


@app.route('/trainer_course', methods = ['GET', 'POST'])
def trainerCourse():
    cur = mysql.connection.cursor()
    # print(session['course_name'])
    cur.execute("SELECT course_overview, rating_count, rating, enrolled"
                " FROM courses WHERE course_name=%s", [session['course_name']])
    d = cur.fetchall()
    # print(d)
    crs = [i for t in d for i in t]
    cur.close()

    if request.method == 'POST':
        page = request.form
        fla = page['img']
        fla1 = page['img']
        if fla != 0:
            cur = mysql.connection.cursor()
            cur.execute("SELECT feedback FROM course_enrolled WHERE course_name = %s", [session['course_name']])
            data8 = cur.fetchall()
            # print(data8)
            feedbacks = [item for t in data8 for item in t]
            # print(feedbacks)

            def percentage(part, whole):
                return 100 * float(part) / float(whole)

            polarity = 0
            positive = 0
            negative = 0
            neutral = 0

            for feedback in feedbacks:
                analysis = TextBlob(feedback)
                polarity += analysis.sentiment.polarity

                if (analysis.sentiment.polarity == 0):
                    neutral += 1
                elif (analysis.sentiment.polarity < 0.00):
                    negative += 1
                elif (analysis.sentiment.polarity > 0.00):
                    positive += 1

            positive = percentage(positive, len(feedbacks))
            negative = percentage(negative, len(feedbacks))
            neutral = percentage(neutral, len(feedbacks))
            polarity = percentage(polarity, len(feedbacks))

            positive = format(positive, '.2f')
            negative = format(negative, '.2f')
            neutral = format(neutral, '.2f')

            if (polarity == 0):
                print("Neutral")
            elif (polarity < 0.00):
                print("Negative")
            if (polarity > 0.00):
                print("Positive")

            labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
                      'Negative [' + str(negative) + '%]']
            sizes = [positive, neutral, negative]
            colors = ['yellowgreen', 'gold', 'red']
            patches, texts = plt.pie(sizes, colors=colors, startangle=90)
            plt.legend(patches, labels, loc="best")
            plt.title(
                'Review of "' + session['course_name'] + '"\n by analyzing ' + str(len(feedbacks)) + ' feedbacks.')
            plt.axis('equal')
            plt.tight_layout()
            # strFile = ("F:/xampp/htdocs/dashboard/images/feedback.png")
            # if os.path.isfile(strFile):
            #     os.remove(strFile)
            # plt.show()
            plt.savefig('F:/xampp/htdocs/dashboard/images/feedback.png')
            # return render_template("lll.html", title = 'Trainer course', crs=crs)
        # if fla1 != 0:
        #     return render_template("lll.html", title='Trainer course101')
    return render_template("trainerCourse.html", title = 'Trainer course', crs=crs)

if __name__ == '__main__':
    app.run(debug=True, port=3333)
