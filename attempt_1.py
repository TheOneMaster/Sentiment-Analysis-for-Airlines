# Imports
import jsonlines
import datetime as dt
import os
import pandas as pd
import sqlite3 as sql
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from palpatine import Sentiment

# Functions


def export_to_sql(input_stuff, output):
    """ 
    Export data to an SQLite database. Creates 4 tables (main, users, replies, hashtags) and uses the attributes \n
    from the JSON file.

    Attributes
    directory - Absolute directory of the folder with the files to be added
    output - Name of the output file (.db)
    """

    con = sql.connect(output)
    vader = SentimentIntensityAnalyzer()
    palpatine = Sentiment()

    # Lists for each table

    # main
    tweet_id, user_id, tweet_time, tweet_text, favourites, retweets, truncated, tweet_lang, sent = [[] for i in range(9)]

    # users
    screen_name, user_created, user_lang, user_desc, verified, followers, default_prof = [[] for i in range(7)]
    default_prof_image, username = [[] for i in range(2)]

    # replies
    reply_id, reply_user = [[] for i in range(2)]

    # hashtags
    hashtags = []

    def get_attrib(file):
        with jsonlines.open(file, mode='r') as main:
            for line in main.iter(allow_none=True, skip_invalid=True, skip_empty=True):

                # Table 1 (main)
                tweet_id.append(line['id_str'])
                user_id.append(line['user']['id_str'])
                tweet_time.append(dt.datetime.strptime(line['created_at'], '%a %b %d %H:%M:%S %z %Y'))
                favourites.append(line['favorite_count'])
                retweets.append(line['retweet_count'])
                truncated.append(line['truncated'])
                text = line['text']
                tweet_text.append(text)
                
                if 'lang' in line:
                    lang = line['lang']
                    tweet_lang.append(lang)
                    if lang == 'en':
                        score = vader.polarity_scores(text)['compound']
                        sent.append(score)
                    else:
                        score = palpatine.basic_polarity(tweet=text, language=lang)
                        if isinstance(score, dict):
                            sent.append(score['compound'])
                        else:
                            sent.append(score)

                else:
                    tweet_lang.append('N/A')
                    sent.append('N/A') 

                # Table 2 (users)
                screen_name.append(line['user']['screen_name'])
                user_created.append(dt.datetime.strptime(line['user']['created_at'], '%a %b %d %H:%M:%S %z %Y').date())
                user_lang.append(line['user']['lang'])
                user_desc.append(line['user']['description'])
                verified.append(line['user']['verified'])
                followers.append(line['user']['followers_count'])
                default_prof.append(line['user']['default_profile'])
                default_prof_image.append(line['user']['default_profile_image'])
                username.append(line['user']['name'])

                # Table 3 (replies)
                reply_id.append(line['in_reply_to_status_id_str'])
                reply_user.append(line['in_reply_to_user_id_str'])

                # Table 4 (hashtags)
                temp_3 = [x['text'] for x in line['entities']['hashtags']]
                if len(temp_3) == 0:
                    hashtags.append(None)
                else:
                    hashtags.append(temp_3)
        
    if os.path.isdir(input_stuff):
        files = [f'{input_stuff}/{x}' for x in os.listdir(input_stuff)]
        for file in files:
            get_attrib(file)
    else:
        get_attrib(input_stuff)

    main_dict = {
        'ID': tweet_id,
        'User ID': user_id,
        'Created At': tweet_time,
        'Text': tweet_text,
        'Language': tweet_lang,
        'Sentiment': sent,
        'Favourites': favourites,
        'Retweets': retweets,
        'Truncated': truncated
    }

    users_dict = {
        'User ID': user_id,
        'Date Created': user_created,
        'Screen Name': screen_name,
        'Name': username,
        'Description': user_desc,
        'Verified': verified,
        'Follower Count': followers,
        'Language': user_lang,
        'Default Profile': default_prof,
        'Default Profile Image': default_prof_image

    }

    temp_1, temp_2, temp_3 = [[] for i in range(3)]

    for x, v, z in zip(tweet_id, reply_id, reply_user):
        if v is None:
            continue
        temp_1.append(x)
        temp_2.append(v)
        temp_3.append(z)

    replies_dict = {
        'ID': temp_1,
        'Reply ID': temp_2,
        'Reply User': temp_3
    }

    temp_1 = [(i, x) for i, v in zip(tweet_id, hashtags) if v is not None for x in v]

    hashtags_dict = {
        'ID': [x[0] for x in temp_1],
        'Hashtag': [x[1] for x in temp_1]
    }

    # Dataframes
    df_main = pd.DataFrame(main_dict)
    df_users = pd.DataFrame(users_dict)
    df_replies = pd.DataFrame(replies_dict)
    df_hashtags = pd.DataFrame(hashtags_dict)

    # Dataframe cleaning
    df_main.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    df_hashtags.drop_duplicates(keep='last', inplace=True)
    df_replies.drop_duplicates(subset=['ID'], keep='last', inplace=True)
    df_users.drop_duplicates(subset=['User ID'], keep='last', inplace=True)

    # Schemas (Don't worry about it, you don't really need to know what this does)
    dtypes_1 = {
        'ID': 'TEXT',
        'User ID': 'TEXT',
        'Created At': 'TEXT',
        'Language': 'TEXT',
        'Text': 'TEXT',
        'Sentiment': 'REAL',
        'Retweets': 'INT',
        'Favourites': 'INT',
        'Truncated': 'INT'
    }

    dtypes_2 = {
        'User ID': 'TEXT',
        'Screen Name': 'TEXT',
        'Name': 'TEXT',
        'Date Created': 'NUMERIC',
        'Description': 'TEXT',
        'Verified': 'INT',
        'Follower Count': 'INT',
        'Language': 'TEXT',
        'Default Profile': 'INT',
        'Default Profile Image': 'INT'
    }

    dtypes_3 = {
        'ID': 'TEXT',
        'Reply ID': 'TEXT',
        'Reply User': 'TEXT'
    }

    dtypes_4 = {
        'ID': 'TEXT',
        'Hashtag': 'TEXT'
    }

    df_main.to_sql('main', con, if_exists='append', index=False, dtype=dtypes_1)
    df_users.to_sql('users', con, if_exists='append', index=False, dtype=dtypes_2)
    df_replies.to_sql('replies', con, if_exists='append', index=False, dtype=dtypes_3)
    df_hashtags.to_sql('hashtag', con, if_exists='append', index=False, dtype=dtypes_4)

    con.close()


if __name__ == '__main__':

    directory = input('Enter the absolute directory of the folder with the files: ')
    output_file = input('Enter the name of the database: ')
    
    export_to_sql(directory, output_file)
    
    print('Done')
