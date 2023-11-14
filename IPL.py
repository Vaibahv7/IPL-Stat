#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)


# In[2]:


matches=pd.read_csv('IPL_Matches_2008_2022.csv')


# In[3]:


matches.head()


# In[4]:


matches.shape


# In[5]:


matches.isnull().sum()


# In[6]:


matches[matches['City'].isnull()==True]['Venue']


# In[7]:


matches.City.fillna('Dubai',inplace=True)


# In[8]:


matches.isnull().sum()


# In[9]:


matches.SuperOver.fillna('N',inplace=True)


# In[10]:


matches['WinningTeam'].fillna('Tie',inplace=True)


# In[11]:


matches.Margin.fillna(0,inplace=True)


# In[12]:


matches.method.fillna('D/L',inplace=True)


# In[13]:


matches.isnull().sum()


# In[14]:


matches['Player_of_Match'].fillna('None',inplace=True)


# In[15]:


matches.shape


# In[16]:


matches.columns


# In[17]:


matches.Season.unique()


# In[18]:


matches.replace({'2009/10':'2010','2007/08':'2008','2020/21':'2020'},inplace=True)


# In[19]:


matches.City.unique()


# In[20]:


matches.replace({'Navi Mumbai':'Mumbai','Bengaluru':'Bangalore'},inplace=True)


# In[21]:


matches.Team1.unique()


# In[22]:


matches.replace({'Rising Pune Supergiant':'Rising Pune Supergiants'},inplace=True)


# In[23]:


matches['WinningTeam'].value_counts()


# In[24]:


matches['Venue'].nunique()


# In[25]:


matches['Venue'].sort_values().unique()


# In[26]:


matches.replace({'Wankhede Stadium, Mumbai':'Wankhede Stadium','Eden Gardens, Kolkata':'Eden Gardens',
                 'Narendra Modi Stadium, Ahmedabad':'Narendra Modi Stadium',
                 'Maharashtra Cricket Association Stadium, Pune':'Maharashtra Cricket Association Stadium',
                 'MA Chidambaram Stadium, Chepauk, Chennai':'MA Chidambaram Stadium',
                 'MA Chidambaram Stadium, Chepauk':'MA Chidambaram Stadium',
                'Punjab Cricket Association Stadium, Mohali':'Punjab Cricket Association Stadium',
                 'Punjab Cricket Association IS Bindra Stadium, Mohali':'Punjab Cricket Association Stadium',
                 'Brabourne Stadium, Mumbai':'Brabourne Stadium',
                 'Vidarbha Cricket Association Stadium, Jamtha':'Vidarbha Cricket Association Stadium',
                 'Arun Jaitley Stadium, Delhi':'Arun Jaitley Stadium',
                 'Dr DY Patil Sports Academy, Mumbai':'Dr DY Patil Sports Academy',
                'M.Chinnaswamy Stadium':'M Chinnaswamy Stadium','MA Chinnaswamy Stadium':'M Chinnaswamy Stadium',
                 'Punjab Cricket Association IS Bindra Stadium':'Punjab Cricket Association Stadium',
                 'Rajiv Gandhi International Stadium, Uppal':'Rajiv Gandhi International Stadium'
                 },inplace=True)


# In[27]:


matches['Venue'].nunique()


# In[28]:


matches.columns


# ### No of seasons completed: 

# In[29]:


print('Season List: ',matches['Season'].unique())
print('No of seasons: ',matches['Season'].nunique())


# ### Winners of each season 

# In[30]:


matches[matches['MatchNumber']=='Final'][['Season','City','Team1','Team2','TossWinner','WinningTeam','Player_of_Match']]


# ### Team with most titles

# In[31]:


titles=matches[matches['MatchNumber']=='Final']['WinningTeam'].value_counts().reset_index()

plt.figure(figsize=(12,4))
sns.barplot(x='WinningTeam',y='count',data=titles)
plt.xticks(fontsize=8)
plt.xlabel('Teams')
plt.ylabel('IPL title Win counts')


# ### Teams played finals

# In[32]:


finals=matches[matches['MatchNumber']=='Final']
most_finals=pd.concat([finals['Team1'],finals['Team2']]).value_counts().reset_index()
winner=finals['WinningTeam'].value_counts().reset_index()
most_finals=most_finals.merge(winner,left_on='index',right_on='WinningTeam',how='outer')
most_finals.replace(np.NaN,0)
most_finals.drop('WinningTeam',axis=1,inplace=True)
most_finals.replace(np.NaN,0,inplace=True)
most_finals.set_index('index',inplace=True)
most_finals.rename({'count_x':'Finals Played','count_y':'Win Count'},axis=1,inplace=True)


# In[33]:


most_finals.plot.bar(width=0.8,color=sns.color_palette("muted", 10))
plt.xlabel('Teams',size=10)
plt.gcf().set_size_inches(8,4)
plt.show()


# In[34]:


most_finals['Win Percentage']=(most_finals['Win Count']/most_finals['Finals Played'])*100


# In[35]:


plt.figure(figsize=(10,5))
ax=sns.barplot(x='index',y='Win Percentage',data=most_finals.reset_index())
plt.xticks(rotation='vertical')
plt.xlabel('Teams',fontsize=(20))
plt.ylabel('Win-Percentage',fontsize=(20))
ax.bar_label(ax.containers[0])
plt.show()


# ### Toss Decision in finals

# In[36]:


toss_finals=finals['TossDecision'].value_counts().reset_index()


# In[37]:


plt.figure(figsize=(4,4))
ax=sns.barplot(x='TossDecision',y='count',data=toss_finals)
ax.bar_label(ax.containers[0])
plt.xlabel('TossDecision',fontsize=(15))
plt.ylabel('TD Count',fontsize=(15))
plt.show()


# In[38]:


toss_winner_final_winner_count=finals[finals['WinningTeam']==finals['TossWinner']]['Season'].count()
toss_winner_final_loser_count=finals[finals['WinningTeam']!=finals['TossWinner']]['Season'].count()


# In[39]:


data=[['Yes',toss_winner_final_winner_count],['No',toss_winner_final_loser_count]]
toss_win_decision=pd.DataFrame(data,columns=['Result','Count'])


# In[40]:


plt.figure(figsize=(4,4))
ax=sns.barplot(x='Result',y='Count',data=toss_win_decision)
ax.bar_label(ax.containers[0])
plt.xlabel('Toss Match Result',fontsize=(15))
plt.ylabel('Result Count',fontsize=(15))
plt.show()


# ### Most MOM's 

# In[41]:


moms = matches['Player_of_Match'].value_counts().head(10).reset_index()


# In[42]:


plt.figure(figsize=(10,5))
mom_plot=sns.barplot(x='Player_of_Match',y='count',data=moms)
plt.xticks(rotation='vertical')
plt.xlabel('Man Of Match',fontsize=(15))
plt.ylabel('Count',fontsize=(15))
mom_plot.bar_label(mom_plot.containers[0])
plt.show()


# ### Team with good win rate

# In[43]:


matches_played=pd.concat([matches['Team1'],matches['Team2']]).value_counts().reset_index()
matches_won=matches['WinningTeam'].value_counts().reset_index()
win_rate=matches_played.merge(matches_won,left_on='index',right_on='WinningTeam',how='outer')
win_rate.set_index('index',inplace=True)
win_rate.drop('WinningTeam',axis=1,inplace=True)
win_rate.rename({'count_x':'Matches-Played','count_y':'Matches-Won'},axis=1,inplace=True)


# In[44]:


win_rate.plot.bar(width=0.8,color=sns.color_palette("muted", 10))
plt.xlabel('Teams',size=10)
plt.gcf().set_size_inches(10,5)
plt.show()


# In[45]:


win_rate['Win-Percentage']=(win_rate['Matches-Won']/win_rate['Matches-Played'])*100
success=win_rate[win_rate['Matches-Won']>50].sort_values(by='Win-Percentage',ascending=False).reset_index()


# In[46]:


plt.figure(figsize=(10,5))
succ_plot=sns.barplot(x='index',y='Win-Percentage',data=success)
plt.xticks(rotation='vertical')
plt.xlabel('Teams',fontsize=(15))
plt.ylabel('Win-Rate',fontsize=(15))
plt.title('Success Rate of teams with 50+ wins',fontsize=(20))
succ_plot.bar_label(succ_plot.containers[0])
plt.show()


# In[47]:


def matchwin_per_venue(venue):
    ven=matches[matches['Venue']==venue]
    top_winner=ven['WinningTeam'].value_counts().reset_index().head(5)
    
    plt.figure(figsize=(7,4))
    wins_venue=sns.barplot(x='WinningTeam',y='count',data=top_winner)
    plt.xticks(rotation='vertical')
    plt.xlabel('Teams',fontsize=(15))
    plt.ylabel('Win-Count',fontsize=(15))
    plt.title(f'TOP 5 WINNING TEAMS ON {venue}',fontsize=(20))
    wins_venue.bar_label(wins_venue.containers[0])
    plt.show()


# In[48]:


matchwin_per_venue('M Chinnaswamy Stadium')


# ## Teams Head to Head

# In[49]:


def team_head_to_head(team1,team2):
    mt1=matches[((matches['Team1']==team1)|(matches['Team2']==team1)) & ((matches['Team1']==team2)|(matches['Team2']==team2))]
    win_count=mt1['WinningTeam'].value_counts().reset_index()
    
    sns.countplot(x='Season',hue='WinningTeam',data=mt1,palette=sns.color_palette("pastel"))
    title=team1 +'VS'+ team2
    plt.title(title)
    plt.xticks(rotation='vertical')
    plt.xlabel('Season',fontsize=(15))
    plt.ylabel('Wins',fontsize=(15))
    
    plt.figure(figsize=(7,5))
    ax=sns.barplot(x='WinningTeam',y='count',data=win_count)
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation='vertical')
    plt.xlabel('Teams',fontsize=(15))
    plt.ylabel('Wins',fontsize=(15))
    
    plt.show()
team_head_to_head('Mumbai Indians','Chennai Super Kings')


# ## Ball-by-Ball Data

# In[50]:


deliveries=pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')


# In[51]:


deliveries.head()


# In[52]:


deliveries.info()


# In[53]:


deliveries.isnull().sum()


# In[54]:


deliveries['extra_type'].fillna('None',inplace=True)


# In[55]:


deliveries['player_out'].fillna('None',inplace=True)


# In[56]:


deliveries['kind'].fillna('None',inplace=True)
deliveries['fielders_involved'].fillna('None',inplace=True)


# In[57]:


top_batsman=deliveries.groupby(['batter'])['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
top_batsman


# In[58]:


plt.figure(figsize=(7,5))
batsman=sns.barplot(x='batter',y='batsman_run',data=top_batsman.head(10))
plt.xticks(rotation='vertical')
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Runs',fontsize=(15))
plt.title('TOP BATSMEN',fontsize=(20))
batsman.bar_label(batsman.containers[0])
plt.show()


# In[59]:


wickets=deliveries[(deliveries['kind'] == 'caught') | (deliveries['kind'] == 'caught and bowled')
           |(deliveries['kind'] == 'bowled')|(deliveries['kind'] == 'stumped')|(deliveries['kind'] == 'lbw')|
           (deliveries['kind'] == 'hit wicket')]


# In[60]:


top_bowlers=wickets.groupby(['bowler'])['isWicketDelivery'].count().reset_index().sort_values(by='isWicketDelivery',ascending=False)


# In[61]:


plt.figure(figsize=(7,5))
bowler=sns.barplot(x='bowler',y='isWicketDelivery',data=top_bowlers.head(10))
plt.xticks(rotation='vertical')
plt.xlabel('Bowler',fontsize=(15))
plt.ylabel('Wickets',fontsize=(15))
plt.title('TOP BOLWERS',fontsize=(20))
bowler.bar_label(bowler.containers[0])
plt.show()


# ### 200+ Scores Batting / Balling First

# In[62]:


scores_200=deliveries.groupby(['ID','innings','BattingTeam'])['total_run'].sum().reset_index()


# In[63]:


scores_200=scores_200[scores_200['total_run']>200]
scores_200


# In[64]:


scores_200_bat_first=scores_200[scores_200['innings']==1]
scores_200_ball_first=scores_200[scores_200['innings']==2]


# In[65]:


scores_200_bat_first_count=scores_200_bat_first['BattingTeam'].value_counts().reset_index()
scores_200_ball_first_count=scores_200_ball_first['BattingTeam'].value_counts().reset_index()


# In[66]:


plt.figure(figsize=(7,5))
sns.barplot(x='BattingTeam',y='count',data=scores_200_bat_first_count)
plt.xticks(rotation='vertical')
plt.xlabel('Teams',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('No Of Times Team Scored 200+ Batting First',size=20)

plt.figure(figsize=(7,5))
sns.barplot(x='BattingTeam',y='count',data=scores_200_ball_first_count)
plt.xticks(rotation='vertical')
plt.xlabel('Teams',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('No Of Times Team Scored 200+ Balling First',size=20)
plt.show()


# ### Successfull 200+ Run Chases

# In[67]:


new=deliveries.merge(matches,left_on='ID',right_on='ID',how='outer')
runchase=new.groupby(['ID','innings','BattingTeam','WinningTeam'])['total_run'].sum().reset_index()
runchase=runchase[runchase['total_run']>200]
runchase=runchase[runchase['innings']==2]
runchase=runchase[runchase['BattingTeam']==runchase['WinningTeam']]
runchase_counts=runchase['WinningTeam'].value_counts().reset_index()


# In[68]:


plt.figure(figsize=(7,5))
sns.barplot(x='WinningTeam',y='count',data=runchase_counts)
plt.xticks(rotation='vertical')
plt.xlabel('Teams',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('No Of Times Team Chased 200+',size=20)
plt.show()


# ### Average Score per Venue

# In[69]:


runs_venue=new.groupby(['innings','Venue'])['total_run'].sum().reset_index()
runs_venue_first_inn=runs_venue[runs_venue['innings']==1]
runs_venue_second_inn=runs_venue[runs_venue['innings']==2]
innings_count=matches['Venue'].value_counts().reset_index()
runs_venue_first_inn=runs_venue_first_inn.merge(innings_count,left_on='Venue',right_on='Venue',how='outer')
runs_venue_second_inn=runs_venue_second_inn.merge(innings_count,left_on='Venue',right_on='Venue',how='outer')
runs_venue_first_inn['Avg_score']=runs_venue_first_inn['total_run']//runs_venue_first_inn['count']
runs_venue_second_inn['Avg_score']=runs_venue_second_inn['total_run']//runs_venue_first_inn['count']
runs_venue_first_inn.rename(columns={'count':'f_inn_count','Avg_score':'f_inn_Avg_score'},inplace=True)
runs_venue_second_inn.rename(columns={'count':'s_inn_count','Avg_score':'s_inn_Avg_score'},inplace=True)
avg_runs_venue=runs_venue_first_inn.merge(runs_venue_second_inn,left_on='Venue',right_on='Venue',how='outer')


# In[70]:


fig,ax=plt.subplots(1,2,figsize=(10,4))
f_inn=sns.barplot(x='innings_x',y='f_inn_Avg_score',data=avg_runs_venue[avg_runs_venue['Venue']=='Wankhede Stadium'],ax=ax[0],width=0.4,color='#81C784')
f_inn.bar_label(f_inn.containers[0])
s_inn=sns.barplot(x='innings_y',y='s_inn_Avg_score',data=avg_runs_venue[avg_runs_venue['Venue']=='Wankhede Stadium'],ax=ax[1],width=0.4,color='#EF5350')
s_inn.bar_label(s_inn.containers[0])


# In[71]:


def Avg_score_per_Venue(venue):
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    f_inn=sns.barplot(x='innings_x',y='f_inn_Avg_score',data=avg_runs_venue[avg_runs_venue['Venue']==venue],ax=ax[0],width=0.4,color='#81C784')
    f_inn.bar_label(f_inn.containers[0])
    s_inn=sns.barplot(x='innings_y',y='s_inn_Avg_score',data=avg_runs_venue[avg_runs_venue['Venue']==venue],ax=ax[1],width=0.4,color='#EF5350')
    s_inn.bar_label(s_inn.containers[0])


# In[72]:


Avg_score_per_Venue('Wankhede Stadium')


# ## Top Run Chases

# In[73]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x=runchase[:5].total_run, y=runchase[:5].index, orient='h')
ax.set_yticklabels(runchase[:5].WinningTeam)
ax.bar_label(ax.containers[0])
plt.show()


# # Player Stats

# ## Batsman

# In[100]:


top_batsman[:5]


# In[74]:


batsman_plt=sns.barplot(x='batter',y='batsman_run',data=top_batsman[:5])
batsman_plt.bar_label(batsman_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Runs',fontsize=(15))
plt.show()


# ## Centuries

# In[75]:


runs_innings=deliveries.groupby(['ID','innings','batter'])['batsman_run'].sum().reset_index()
cent=runs_innings[(runs_innings['batsman_run']>=100) & (runs_innings['batsman_run']<150)]
centuries=cent['batter'].value_counts().reset_index(name='Centuries')


# In[76]:


cent_plt=sns.barplot(x='batter',y='Centuries',data=centuries[:5])
cent_plt.bar_label(cent_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Centuries',fontsize=(15))


# ## Highest Individual Score

# In[77]:


top_scorer=runs_innings.sort_values(by='batsman_run',ascending=False)
top_scorer


# In[78]:


high_ind_plt=sns.barplot(x='batter',y='batsman_run',data=top_scorer[:5])
high_ind_plt.bar_label(high_ind_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Individual Score',fontsize=(15))


# ## Powerplay Batting Master

# In[79]:


pow_batsman_scores=deliveries[deliveries['overs']<6].groupby(['batter'])['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
pow_wickets=deliveries[deliveries['overs']<6]['player_out'].value_counts().reset_index()
pow_wickets.drop(0,inplace=True)
pow_batsman_scores=pow_batsman_scores.merge(pow_wickets,left_on='batter',right_on='player_out',how='outer')
pow_batsman_scores.drop(columns=['player_out'],axis=1,inplace=True)
pow_batsman_scores.fillna(0,inplace=True)
pow_batsman_scores['Average']=np.round(pow_batsman_scores['batsman_run']/pow_batsman_scores['count'],2)
pow_hitter=pow_batsman_scores[(pow_batsman_scores['count']!=0) & (pow_batsman_scores['batsman_run']>1000)].sort_values(by='Average',ascending=False)


# In[80]:


pow_hit_plt=sns.barplot(x='batter',y='Average',data=pow_hitter[:5])
pow_hit_plt.bar_label(pow_hit_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Powerplay Average',fontsize=(15))


# ## 1st inning Powerplay Master

# In[81]:


pow_first_inn=deliveries[(deliveries['innings']==1)&(deliveries['overs']<6) ].groupby(['batter'])['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
pow_first_inn_wickets=deliveries[(deliveries['overs']<6)&(deliveries['innings']==1)]['player_out'].value_counts().reset_index()
pow_first_inn_wickets.drop(0,inplace=True)
pow_first_inn=pow_first_inn.merge(pow_first_inn_wickets,left_on='batter',right_on='player_out',how='outer')
pow_first_inn.drop(columns=['player_out'],inplace=True)
pow_first_inn.fillna(0,inplace=True)
pow_first_inn['Averae']=np.round(pow_first_inn['batsman_run']/pow_first_inn['count'],2)
pow_first_inn_hitter=pow_first_inn[(pow_first_inn['count']!=0) & (pow_first_inn['batsman_run']>1000)].sort_values(by='Averae',ascending=False)


# In[82]:


pow_f_hit_plt=sns.barplot(x='batter',y='Averae',data=pow_first_inn_hitter)
pow_f_hit_plt.bar_label(pow_f_hit_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('1st inning Powerplay Average',fontsize=(15))
plt.show()


# ## 2nd Inning Powerplay Master

# In[83]:


pow_second_inn=deliveries[(deliveries['innings']==2)&(deliveries['overs']<6) ].groupby(['batter'])['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
pow_second_inn_wickets=deliveries[(deliveries['overs']<6)&(deliveries['innings']==2)]['player_out'].value_counts().reset_index()
pow_second_inn_wickets.drop(0,inplace=True)
pow_second_inn=pow_second_inn.merge(pow_second_inn_wickets,left_on='batter',right_on='player_out',how='outer')
pow_second_inn.drop(columns=['player_out'],inplace=True)
pow_second_inn.fillna(0,inplace=True)
pow_second_inn['Averae']=np.round(pow_second_inn['batsman_run']/pow_second_inn['count'],2)
pow_second_inn_hitter=pow_first_inn[(pow_second_inn['count']!=0) & (pow_second_inn['batsman_run']>1000)].sort_values(by='Averae',ascending=False)


# In[84]:


pow_s_hit_plt=sns.barplot(x='batter',y='Averae',data=pow_second_inn_hitter[:5])
pow_s_hit_plt.bar_label(pow_s_hit_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('2nd inning Powerplay Average',fontsize=(15))


# ## Death Over Hitter

# In[85]:


death_scores=deliveries[deliveries['overs']>=15].groupby('batter')['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
death_wickets=deliveries[deliveries['overs']>=15]['player_out'].value_counts().reset_index()
death_wickets.drop(0,inplace=True)
death_scores=death_scores.merge(death_wickets,left_on='batter',right_on='player_out',how='outer')
death_scores.drop(columns=['player_out'],inplace=True)
death_scores.fillna(0,inplace=True)
death_scores['Average']=np.round(death_scores['batsman_run']/death_scores['count'],2)
death_scores=death_scores[(death_scores['batsman_run']>1000)].sort_values(by='Average',ascending=False)


# In[86]:


dth_plt=sns.barplot(x='batter',y='Average',data=death_scores[:5])
dth_plt.bar_label(dth_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Death Over Average',fontsize=(15))
plt.show()


# ## PowerPlay Strike Rate

# In[87]:


pow_balls=deliveries[deliveries['overs']<6].groupby('batter')['ballnumber'].count().reset_index(name='balls_played')
pow_sr=pow_batsman_scores.merge(pow_balls,left_on='batter',right_on='batter',how='outer')
pow_sr['S/R']=np.round((pow_sr['batsman_run']/pow_sr['balls_played'])*100,2)
pow_sr=pow_sr[pow_sr['batsman_run']>1500].sort_values(by='S/R',ascending=False)


# In[88]:


plt.figure(figsize=(8,5))
psr_plt=sns.barplot(x='batter',y='S/R',data=pow_sr[:7])
psr_plt.bar_label(psr_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Powerplay Strike Rate',fontsize=(15))
plt.show()


# ## Death Over StrikeRate

# In[89]:


death_balls=deliveries[deliveries['overs']>=15].groupby('batter')['ballnumber'].count().reset_index(name='balls_played')
death_sr=death_scores.merge(death_balls,left_on='batter',right_on='batter',how='outer')
death_sr['S/R']=np.round((death_sr['batsman_run']/death_sr['balls_played'])*100,2)
death_sr=death_sr.sort_values(by='S/R',ascending=False)


# In[90]:


plt.figure(figsize=(8,4))
dth_plt=sns.barplot(x='batter',y='S/R',data=death_sr[:7])
dth_plt.bar_label(dth_plt.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Death Overs Strike Rate',fontsize=(15))


# ## Hihest Batting Average and Strike Rate

# In[91]:


batsman_wickets=deliveries['player_out'].value_counts().reset_index(name='wicket_count')
batsman_wickets.drop(0,inplace=True)
batsman_st=top_batsman.merge(batsman_wickets,left_on='batter',right_on='player_out',how='outer')
batsman_st.drop(columns=['player_out'],inplace=True,axis=1)
balls_played=deliveries.groupby('batter')['ballnumber'].count().reset_index(name='balls_played')
batsman_st=batsman_st.merge(balls_played,left_on='batter',right_on='batter',how='outer')
batsman_st['Avg']=np.round(batsman_st['batsman_run']/batsman_st['wicket_count'],2)
batsman_st['S/R']=np.round((batsman_st['batsman_run']/batsman_st['balls_played'])*100,2)
batsman_st.fillna(0,inplace=True)


# In[92]:


avg=sns.barplot(x='batter',y='Avg',data=batsman_st[batsman_st['batsman_run']>4000].sort_values(by='Avg',ascending=False)[:5])
avg.bar_label(avg.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Average',fontsize=(15))


# In[93]:


sr=sns.barplot(x='batter',y='S/R',data=batsman_st[batsman_st['batsman_run']>4000].sort_values(by='S/R',ascending=False)[:5])
sr.bar_label(sr.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Strike Rate',fontsize=(15))


# ## 1st Inning Avg and SR

# In[94]:


first_inning_bat_st=deliveries[deliveries['innings']==1].groupby('batter')['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
first_inning_wickets=deliveries[deliveries['innings']==1]['player_out'].value_counts().reset_index(name='wicket_count')
first_inning_wickets.drop(0,inplace=True)
first_inning_bat_st=first_inning_bat_st.merge(first_inning_wickets,left_on='batter',right_on='player_out',how='outer')
first_inning_bat_st.drop(columns=['player_out'],axis=1,inplace=True)
first_inn_ball_played=deliveries[deliveries['innings']==1].groupby('batter')['ballnumber'].count().reset_index(name='balls_played')
first_inning_bat_st=first_inning_bat_st.merge(first_inn_ball_played,left_on='batter',right_on='batter',how='outer')
first_inning_bat_st['Avg']=np.round(first_inning_bat_st['batsman_run']/first_inning_bat_st['wicket_count'],2)
first_inning_bat_st['S/R']=np.round((first_inning_bat_st['batsman_run']/first_inning_bat_st['balls_played'])*100,2)
first_inning_bat_st.fillna(0,inplace=True)


# In[95]:


fig,ax=plt.subplots(2,1,figsize=(7,8))

avg=sns.barplot(x='batter',y='Avg',data=first_inning_bat_st[first_inning_bat_st['batsman_run']>2000].sort_values(by='Avg',ascending=False)[:5],ax=ax[0])
avg.bar_label(avg.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Average Rate',fontsize=(15))

sr=sns.barplot(x='batter',y='S/R',data=first_inning_bat_st[first_inning_bat_st['batsman_run']>2000].sort_values(by='S/R',ascending=False)[:5],ax=ax[1])
sr.bar_label(sr.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Strike Rate',fontsize=(15))
plt.show()


# ## 2nd Inning Avg and SR

# In[96]:


second_inning_bat_st=deliveries[deliveries['innings']==2].groupby('batter')['batsman_run'].sum().reset_index().sort_values(by='batsman_run',ascending=False)
second_inning_wickets=deliveries[deliveries['innings']==2]['player_out'].value_counts().reset_index(name='wicket_count')
second_inning_wickets.drop(0,inplace=True)
second_inning_bat_st=second_inning_bat_st.merge(second_inning_wickets,left_on='batter',right_on='player_out',how='outer')
second_inning_bat_st.drop(columns=['player_out'],axis=1,inplace=True)
second_inn_ball_played=deliveries[deliveries['innings']==2].groupby('batter')['ballnumber'].count().reset_index(name='balls_played')
second_inning_bat_st=second_inning_bat_st.merge(second_inn_ball_played,left_on='batter',right_on='batter',how='outer')
second_inning_bat_st['Avg']=np.round(second_inning_bat_st['batsman_run']/second_inning_bat_st['wicket_count'],2)
second_inning_bat_st['S/R']=np.round((second_inning_bat_st['batsman_run']/second_inning_bat_st['balls_played'])*100,2)
second_inning_bat_st.fillna(0,inplace=True)


# In[97]:


fig,ax=plt.subplots(2,1,figsize=(7,8))

avg=sns.barplot(x='batter',y='Avg',data=second_inning_bat_st[second_inning_bat_st['batsman_run']>2000].sort_values(by='Avg',ascending=False)[:5],ax=ax[0])
avg.bar_label(avg.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Average Rate',fontsize=(15))

sr=sns.barplot(x='batter',y='S/R',data=second_inning_bat_st[second_inning_bat_st['batsman_run']>2000].sort_values(by='S/R',ascending=False)[:5],ax=ax[1])
sr.bar_label(sr.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Strike Rate',fontsize=(15))
plt.show()


# ## Batsman Per Venue

# In[98]:


batsman_ven=new.groupby(['Venue','batter'])['batsman_run'].sum().reset_index()
bat_v_balls=new.groupby(['Venue','batter'])['ballnumber'].count().reset_index(name='balls_played')
b_v_wic=new.groupby('Venue')['player_out'].value_counts().reset_index(name='wicket_count')
b_v_wic=b_v_wic[b_v_wic['player_out']!='None']
batsman_ven=batsman_ven.merge(bat_v_balls,left_on=['Venue','batter'],right_on=['Venue','batter'],how='outer')
batsman_ven=batsman_ven.merge(b_v_wic,left_on=['Venue','batter'],right_on=['Venue','player_out'],how='outer')
batsman_ven.drop(columns=['player_out'],axis=1,inplace=True)
batsman_ven.fillna(0,inplace=True)
batsman_ven['Avg']=np.round(batsman_ven['batsman_run']/batsman_ven['wicket_count'],2)
batsman_ven['S/R']=np.round((batsman_ven['batsman_run']/batsman_ven['balls_played'])*100,2)


# In[99]:


def batsman_venue_st(venue,batsman):
    dt=batsman_ven[(batsman_ven['Venue']==venue) & (batsman_ven['batter']==batsman)]
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    av=sns.barplot(x='batter',y='Avg',data=dt,ax=ax[0],width=0.4,color='#81C784')
    av.bar_label(av.containers[0])
    plt.xlabel('Batsman',fontsize=(10))
    plt.ylabel('Average',fontsize=(10))
    
    sr=sns.barplot(x='batter',y='S/R',data=dt,ax=ax[1],width=0.4,color='#EF5350')
    sr.bar_label(sr.containers[0])
    plt.xlabel('Batsman',fontsize=(10))
    plt.ylabel('Strike Rate',fontsize=(10))


# In[100]:


batsman_venue_st('M Chinnaswamy Stadium','V Kohli')


# ## Orange Cap Holders per Season

# In[102]:


orangecap=new.groupby(['Season','batter'])['batsman_run'].sum().reset_index().sort_values(by=['Season','batsman_run'],ascending=[True,False])
orangecap=orangecap.drop_duplicates(subset='Season')
orange_count=orangecap['batter'].value_counts().reset_index(name='Orange Caps')


# In[103]:


oc=sns.barplot(x='batter',y='Orange Caps',data=orange_count[orange_count['Orange Caps']>1])
oc.bar_label(oc.containers[0])
plt.xlabel('Batsman',fontsize=(15))
plt.ylabel('Orange Caps more than once',fontsize=(15))
plt.show()


# ## batsman runs per season

# In[104]:


batsman_season=new.groupby(['batter','Season'])['batsman_run'].sum().reset_index()


# In[105]:


def batsman_per_season(batsman):
    plt.figure(figsize=(10,6))
    batsman_season[batsman_season['batter']==batsman].plot('Season','batsman_run',color='green',marker='o')
    fig=plt.gcf()
    fig.set_size_inches(10,4)
    plt.title(f'{batsman} Total Runs Per Season')
    plt.show()


# In[106]:


batsman_per_season('RG Sharma')


# # Boundaries 

# In[107]:


fours=new[(new['batsman_run']==4)].groupby('Season')['batsman_run'].count().reset_index(name='fours')
sixes=new[(new['batsman_run']==6)].groupby('Season')['batsman_run'].count().reset_index(name='sixes')
boundaries=fours.merge(sixes,left_on='Season',right_on='Season',how='outer')


# In[108]:


boundaries.plot(x='Season',y=['fours','sixes'],color=['#B71C1C','#772272'])
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.legend(loc='upper right')
plt.xlabel('Season',fontsize=15)
plt.ylabel('Boundaries',fontsize=15)

plt.title("Boundaries Per Season",size=20)
plt.show()


# In[118]:


fours_venue=new[(new['batsman_run']==4)].groupby('Venue')['batsman_run'].count().reset_index(name='fours').sort_values(by='fours',ascending=False)
sixes_venue=new[(new['batsman_run']==6)].groupby('Venue')['batsman_run'].count().reset_index(name='sixes').sort_values(by='sixes',ascending=False)
boundaries_venue=fours_venue.merge(sixes_venue,left_on='Venue',right_on='Venue',how='outer')
match_venue=matches['Venue'].value_counts().reset_index(name='matches')
boundaries_venue=boundaries_venue.merge(match_venue,left_on='Venue',right_on='Venue',how='outer')
boundaries_venue['fours_avg']=boundaries_venue['fours']//boundaries_venue['matches']
boundaries_venue['sixes_avg']=boundaries_venue['sixes']//boundaries_venue['matches']


# In[121]:


def venue_boundaries(venue):
    
    dt=boundaries_venue[(boundaries_venue['Venue']==venue)]
    print(dt)
    fig,ax=plt.subplots(2,2,figsize=(10,8))
    av=sns.barplot(x='Venue',y='fours',data=dt,ax=ax[0,0],width=0.2,color='#81C784')
    av.bar_label(av.containers[0])
    plt.xlabel('Venue',fontsize=(10))
    plt.ylabel('Fours',fontsize=(10))
    
    sr=sns.barplot(x='Venue',y='sixes',data=dt,ax=ax[0,1],width=0.2,color='#81C784')
    sr.bar_label(sr.containers[0])
    plt.xlabel('Venue',fontsize=(10))
    plt.ylabel('Sixes',fontsize=(10))
    
    f_av=sns.barplot(x='Venue',y='fours_avg',data=dt,ax=ax[1,0],width=0.2,color='#EF5350')
    f_av.bar_label(f_av.containers[0])
    plt.xlabel('Venue',fontsize=(10))
    plt.ylabel('Avg Fours',fontsize=(10))
    
    s_av=sns.barplot(x='Venue',y='sixes_avg',data=dt,ax=ax[1,1],width=0.2,color='#EF5350')
    s_av.bar_label(s_av.containers[0])
    plt.xlabel('Venue',fontsize=(10))
    plt.ylabel('Avg Sixes',fontsize=(10))
    


# In[122]:


venue_boundaries('M Chinnaswamy Stadium')


# In[123]:


x=boundaries_venue[boundaries_venue['matches']>50].sort_values(by='fours_avg',ascending=False)
y=boundaries_venue[boundaries_venue['matches']>50].sort_values(by='sixes_avg',ascending=False)
fig,ax=plt.subplots(2,1,figsize=(12,8))
xf=sns.barplot(x='Venue',y='fours_avg',data=x[:5],ax=ax[0],width=0.4)
xf.bar_label(xf.containers[0])
plt.xticks(rotation='vertical')
plt.xlabel('Venue',fontsize=(15))
plt.ylabel('Fours Average',fontsize=(15))

sf=sns.barplot(x='Venue',y='sixes_avg',data=y[:5],ax=ax[1],width=0.4)
sf.bar_label(sf.containers[0])
plt.xticks(rotation='vertical')
plt.xlabel('Venue',fontsize=(15))
plt.ylabel('Sixes Average',fontsize=(15))
plt.show()


# ### Boundaries per batsman

# In[124]:


b_bound=deliveries[deliveries['batsman_run']==4].groupby('batter')['batsman_run'].count().reset_index(name='fours')
b_six=deliveries[deliveries['batsman_run']==6].groupby('batter')['batsman_run'].count().reset_index(name='sixes')
b_bound=b_bound.merge(b_six,left_on='batter',right_on='batter',how='outer')
b_matches=deliveries.groupby('batter')['ID'].nunique().reset_index(name='matches').sort_values(by='matches',ascending=False)
b_bound=b_bound.merge(b_matches,left_on='batter',right_on='batter',how='outer')
b_bound.fillna(0,inplace=True)
b_bound['fours Avg']=np.ceil(b_bound['fours']/b_bound['matches'])
b_bound['sixes Avg']=np.ceil(b_bound['sixes']/b_bound['matches'])


# In[125]:


fig,ax=plt.subplots(2,2,figsize=(18,12))
xf=sns.barplot(x='fours',y='batter',data=b_bound.sort_values(by='fours',ascending=False)[:5],ax=ax[0,0],width=0.4,orient='h',color='#80DEEA')
xf.bar_label(xf.containers[0])
ax[0,0].set_ylabel('')
ax[0,0].set_title('Highest Fours',fontsize=(15))

xf=sns.barplot(x='sixes',y='batter',data=b_bound.sort_values(by='sixes',ascending=False)[:5],ax=ax[0,1],width=0.4,orient='h',color='#FFF176')
xf.bar_label(xf.containers[0])
ax[0,1].set_ylabel('')
ax[0,1].set_title('Highest Sixes',fontsize=(15))

xf=sns.barplot(x='fours Avg',y='batter',data=b_bound.sort_values(by='fours Avg',ascending=False)[:5],ax=ax[1,0],width=0.4,orient='h',color='#81C784')
xf.bar_label(xf.containers[0])
ax[1,0].set_ylabel('')
ax[1,0].set_title('Highest Fours Avg',fontsize=(15))

xf=sns.barplot(x='sixes Avg',y='batter',data=b_bound.sort_values(by='sixes Avg',ascending=False)[:5],ax=ax[1,1],width=0.4,orient='h',color='#EF5350')
xf.bar_label(xf.containers[0])
ax[1,1].set_ylabel('')
ax[1,1].set_title('Highest Sixes Avg',fontsize=(15))


# # Bowler Stats

# In[127]:


wickets=deliveries[(deliveries['kind'] == 'caught') | (deliveries['kind'] == 'caught and bowled')
           |(deliveries['kind'] == 'bowled')|(deliveries['kind'] == 'stumped')|(deliveries['kind'] == 'lbw')|
           (deliveries['kind'] == 'hit wicket')]
bowler=wickets.groupby(['bowler'])['isWicketDelivery'].count().reset_index().sort_values(by='isWicketDelivery',ascending=False)
wicket_type=deliveries['kind'].value_counts().reset_index(name='Wicket Type')
wicket_type.drop(0,inplace=True)


# In[128]:


wt=wicket_type[:6]
plt.figure(figsize=(6,6))
plt.pie(wt['Wicket Type'],labels=wt['kind'],autopct='%1.1f%%',
        colors=['#66CC99','#CC9999','#CCFF99','#B3E5FC','#B71C1C','#E59866'])
plt.show()


# In[129]:


plt.figure(figsize=(8,4))
bw=sns.barplot(x='isWicketDelivery',y='bowler',data=bowler[:5],width=0.4,orient='h',color='#80DEEA')
bw.bar_label(bw.containers[0])
plt.xlabel('Wickets',fontsize=(15))
plt.ylabel('Bowlers',fontsize=(15))
plt.title('Top Wicket Takers',fontsize=(20))
plt.show()


# In[130]:


overs=deliveries.groupby(['bowler'])['ballnumber'].count().reset_index().sort_values(by='ballnumber',ascending=False)
overs['overs']=overs['ballnumber']//6
bowler=bowler.merge(overs,left_on='bowler',right_on='bowler',how='outer')
bowler.fillna(0,inplace=True)
bowler.drop(columns=['ballnumber'],inplace=True,axis=1)
bow_runs=deliveries[deliveries['extra_type'] !='penalty'].groupby('bowler')['total_run'].sum().reset_index(name='Runs Condceded')
bowler=bowler.merge(bow_runs,left_on='bowler',right_on='bowler',how='outer')


# In[131]:


bowler['Economy']=np.round(bowler['Runs Condceded']/bowler['overs'],2)
bowler['Avg']=np.round(bowler['Runs Condceded']/bowler['isWicketDelivery'],2)
bowler['S/r']=np.round((bowler['overs']*6)/bowler['isWicketDelivery'],2)


# In[132]:


bowler.fillna(0,inplace=True)


# In[133]:


x=bowler[bowler['isWicketDelivery']>100].sort_values(by='Avg')
y=bowler[bowler['isWicketDelivery']>100].sort_values(by='Economy')
z=bowler[bowler['isWicketDelivery']>100].sort_values(by='S/r')

fig,ax=plt.subplots(3,1,figsize=(8,12))

xplt=sns.barplot(x='Avg',y='bowler',data=x[:5],orient='h',width=0.4,ax=ax[0],color='#81C784')
xplt.bar_label(xplt.containers[0])
ax[0].set_ylabel('')
ax[0].set_xlabel('')
ax[0].set_title('Best Average',fontsize=(15))

yplt=sns.barplot(x='Economy',y='bowler',data=y[:5],orient='h',width=0.4,ax=ax[1],color='#EF5350')
yplt.bar_label(yplt.containers[0])
ax[1].set_ylabel('')
ax[1].set_xlabel('')
ax[1].set_title('Best Economy',fontsize=(15))

zplt=sns.barplot(x='S/r',y='bowler',data=z[:5],orient='h',width=0.4,ax=ax[2],color='#FFF176')
zplt.bar_label(zplt.containers[0])
ax[2].set_ylabel('')
ax[2].set_xlabel('')
ax[2].set_title('Best Strike Rate',fontsize=(15))

plt.show()


# ## Highest Wicket Taker in match

# In[134]:


best_bow_figss=wickets.groupby(['ID','bowler'])['isWicketDelivery'].count().reset_index().sort_values(by='isWicketDelivery',ascending=False)
runs_match=deliveries[deliveries['extra_type'] !='penalty'].groupby(['ID','bowler'])['total_run'].sum().reset_index(name='Runs Condceded')
best_bow_figss=best_bow_figss.merge(runs_match,left_on=['ID','bowler'],right_on=['ID','bowler'],how='outer')
best_bow_figss.fillna(0,inplace=True)
best_bow_figss['isWicketDelivery']=best_bow_figss['isWicketDelivery'].astype(int)
best_bow_figss['Best Fig']=best_bow_figss['isWicketDelivery'].astype(str) + "/" + best_bow_figss['Runs Condceded'].astype(str)


# ## Purple Cap Holder

# In[135]:


wickets=new[(new['kind'] == 'caught') | (new['kind'] == 'caught and bowled')
           |(new['kind'] == 'bowled')|(new['kind'] == 'stumped')|(new['kind'] == 'lbw')|
           (new['kind'] == 'hit wicket')]


# In[136]:


purple_cap=wickets.groupby(['Season','bowler'])['isWicketDelivery'].count().reset_index().sort_values(by=['Season','isWicketDelivery'],ascending=[True,False])
purple_cap=purple_cap.drop_duplicates(subset='Season')


# In[137]:


plt.figure(figsize=(10,5))
pur_plt = sns.barplot(x=purple_cap.index, y=purple_cap.isWicketDelivery,color='purple')
pur_plt.set_xticklabels(purple_cap.bowler)
pur_plt.bar_label(pur_plt.containers[0])
plt.xticks(rotation='vertical')
plt.xlabel('Bowlers',fontsize=(15))
plt.ylabel('Wickets',fontsize=(15))
plt.show()


# # Bowlers at Venue

# In[138]:


bow_venue=wickets.groupby(['Venue','bowler'])['isWicketDelivery'].count().reset_index()


# In[139]:


def bowlers_top_wickets_venue(bowler):
    dt=bow_venue[bow_venue['bowler']==bowler].sort_values(by='isWicketDelivery',ascending=False)
    dt_plt=sns.barplot(y='Venue',x='isWicketDelivery',data=dt[:5],orient='h',color='#00827F')
    dt_plt.bar_label(dt_plt.containers[0])
    plt.xlabel('Wickets',fontsize=(15))
    plt.ylabel('Venue',fontsize=(15))
    plt.title(f'{bowler}\'s Top 5 Wicket Taking Venue',fontsize=(20))


# In[140]:


bowlers_top_wickets_venue('JJ Bumrah')


# In[141]:


def venue_bowlers(venue):
    dt=bow_venue[bow_venue['Venue']==venue].sort_values(by='isWicketDelivery',ascending=False)
    dt_plt=sns.barplot(y='bowler',x='isWicketDelivery',data=dt[:5],orient='h',color='#08A04B')
    dt_plt.bar_label(dt_plt.containers[0])
    plt.xlabel('Wickets',fontsize=(15))
    plt.ylabel('Bowler',fontsize=(15))
    plt.title(f'Top 5 Wicket Taker at {venue}',fontsize=(20))


# In[142]:


venue_bowlers('Eden Gardens')


# ## Wickets Against Team

# In[143]:


bow_opp=wickets.groupby(['BattingTeam','bowler'])['isWicketDelivery'].count().reset_index().sort_values(by=['BattingTeam','isWicketDelivery'],ascending=[True,False])
bow_over_opp=deliveries.groupby(['BattingTeam','bowler'])['ballnumber'].count().reset_index()
bow_over_opp['overs']=bow_over_opp['ballnumber']//6
bow_runs_opp=deliveries[deliveries['extra_type'] !='penalty'].groupby(['BattingTeam','bowler'])['total_run'].sum().reset_index(name='Runs_Condceded')
bow_opp=bow_opp.merge(bow_over_opp,left_on=['BattingTeam','bowler'],right_on=['BattingTeam','bowler'],how='outer')
bow_opp=bow_opp.merge(bow_runs_opp,left_on=['BattingTeam','bowler'],right_on=['BattingTeam','bowler'],how='outer')
bow_opp.fillna(0,inplace=True)
bow_opp['isWicketDelivery']=bow_opp['isWicketDelivery'].astype(int)
bow_opp['Avg']=np.round(bow_opp['Runs_Condceded']/bow_opp['isWicketDelivery'],2)
bow_opp['Economy']=np.round(bow_opp['Runs_Condceded']/bow_opp['overs'],2)
bow_opp['S/R']=np.round(bow_opp['ballnumber']/bow_opp['isWicketDelivery'],2)
bow_opp.fillna(0,inplace=True)


# In[144]:


def bow_opp_st(opp):
    ov=bow_opp[bow_opp['overs']>25]
    x=ov[ov['BattingTeam']==opp]
    y=ov[ov['BattingTeam']==opp].sort_values(by='Avg')
    z=ov[ov['BattingTeam']==opp].sort_values(by='Economy')
    m=ov[ov['BattingTeam']==opp].sort_values(by='S/R')

    fig,ax=plt.subplots(4,1,figsize=(10,14))
    
    x_p=sns.barplot(x='isWicketDelivery',y='bowler',data=x[:5],orient='h',width=0.4,ax=ax[0],color='#81C784')
    x_p.bar_label(x_p.containers[0])
    ax[0].set_ylabel('')
    ax[0].set_xlabel('')
    ax[0].set_title('Top Wicket Takers',fontsize=(15))
    
    y_p=sns.barplot(x='Avg',y='bowler',data=y[:5],orient='h',width=0.4,ax=ax[1],color='#EDE275')
    y_p.bar_label(y_p.containers[0])
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    ax[1].set_title('Best Average',fontsize=(15))
    
    z_p=sns.barplot(x='Economy',y='bowler',data=z[:5],orient='h',width=0.4,ax=ax[2],color='#786D5F')
    z_p.bar_label(z_p.containers[0])
    ax[2].set_ylabel('')
    ax[2].set_xlabel('')
    ax[2].set_title('Best Economy',fontsize=(15))
    
    m_p=sns.barplot(x='S/R',y='bowler',data=m[:5],orient='h',width=0.4,ax=ax[3],color='#E66C2C')
    m_p.bar_label(m_p.containers[0])
    ax[3].set_ylabel('')
    ax[3].set_xlabel('')
    ax[3].set_title('Best Strike Rate',fontsize=(15))
    


# In[145]:


bow_opp_st('Chennai Super Kings')


# ## Power Play Bowler

# In[162]:


pow_bow=wickets[wickets['overs']<6].groupby(['bowler'])['isWicketDelivery'].count().reset_index().sort_values(by='isWicketDelivery',ascending=False)


# In[163]:


pow_bow_overs=deliveries[deliveries['overs']<6].groupby(['bowler'])['ballnumber'].count().reset_index()
pow_bow_runs=deliveries[(deliveries['overs']<6) & (deliveries['extra_type']!='penalty')].groupby(['bowler'])['total_run'].sum().reset_index()


# In[164]:


pow_bow_overs['overs']=pow_bow_overs['ballnumber']//6


# In[167]:


pow_bow=pow_bow.merge(pow_bow_overs,left_on='bowler',right_on='bowler',how='outer')
pow_bow=pow_bow.merge(pow_bow_runs,left_on='bowler',right_on='bowler',how='outer')


# In[170]:


pow_bow.fillna(0,inplace=True)
pow_bow['isWicketDelivery']=pow_bow['isWicketDelivery'].astype(int)


# In[172]:


pow_bow['Avg']=np.round(pow_bow['total_run']/pow_bow['isWicketDelivery'],2)
pow_bow['Economy']=np.round(pow_bow['total_run']/pow_bow['overs'],2)
pow_bow['S/R']=np.round(pow_bow['ballnumber']/pow_bow['isWicketDelivery'],2)


# In[181]:


def bow_pow_st():
    ov=pow_bow[pow_bow['overs']>60]
    x=ov.sort_values(by='isWicketDelivery',ascending=False)
    y=ov.sort_values(by='Avg')
    z=ov.sort_values(by='Economy')
    m=ov.sort_values(by='S/R')

    fig,ax=plt.subplots(4,1,figsize=(10,15))
    
    x_p=sns.barplot(x='isWicketDelivery',y='bowler',data=x[:5],orient='h',width=0.4,ax=ax[0],color='#81C784')
    x_p.bar_label(x_p.containers[0])
    ax[0].set_ylabel('')
    ax[0].set_xlabel('')
    ax[0].set_title('Top Powerplay Strikers',fontsize=(15))
    
    y_p=sns.barplot(x='Avg',y='bowler',data=y[:5],orient='h',width=0.4,ax=ax[1],color='#EDE275')
    y_p.bar_label(y_p.containers[0])
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    ax[1].set_title('Best Powerplay Average',fontsize=(15))
    
    z_p=sns.barplot(x='Economy',y='bowler',data=z[:5],orient='h',width=0.4,ax=ax[2],color='#786D5F')
    z_p.bar_label(z_p.containers[0])
    ax[2].set_ylabel('')
    ax[2].set_xlabel('')
    ax[2].set_title('Best Powerplay Economy',fontsize=(15))
    
    m_p=sns.barplot(x='S/R',y='bowler',data=m[:5],orient='h',width=0.4,ax=ax[3],color='#E66C2C')
    m_p.bar_label(m_p.containers[0])
    ax[3].set_ylabel('')
    ax[3].set_xlabel('')
    ax[3].set_title('Best Powerplay Strike Rate',fontsize=(15))
    


# In[182]:


bow_pow_st()


# ## Death Over Specialist

# In[183]:


dth_bow=wickets[wickets['overs']>=15].groupby(['bowler'])['isWicketDelivery'].count().reset_index().sort_values(by='isWicketDelivery',ascending=False)
dth_bow_overs=deliveries[deliveries['overs']>=15].groupby(['bowler'])['ballnumber'].count().reset_index()
dth_bow_runs=deliveries[(deliveries['overs']>=15) & (deliveries['extra_type']!='penalty')].groupby(['bowler'])['total_run'].sum().reset_index()
dth_bow_overs['overs']=dth_bow_overs['ballnumber']//6

dth_bow=dth_bow.merge(dth_bow_overs,left_on='bowler',right_on='bowler',how='outer')
dth_bow=dth_bow.merge(dth_bow_runs,left_on='bowler',right_on='bowler',how='outer')
dth_bow.fillna(0,inplace=True)
dth_bow['isWicketDelivery']=dth_bow['isWicketDelivery'].astype(int)


# In[185]:


dth_bow['Avg']=np.round(dth_bow['total_run']/dth_bow['isWicketDelivery'],2)
dth_bow['Economy']=np.round(dth_bow['total_run']/dth_bow['overs'],2)
dth_bow['S/R']=np.round(dth_bow['ballnumber']/dth_bow['isWicketDelivery'],2)


# In[191]:


def bow_dth_st():
    ov=dth_bow[dth_bow['overs']>60]
    x=ov.sort_values(by='isWicketDelivery',ascending=False)
    y=ov.sort_values(by='Avg')
    z=ov.sort_values(by='Economy')
    m=ov.sort_values(by='S/R')

    fig,ax=plt.subplots(4,1,figsize=(10,15))
    
    x_p=sns.barplot(x='isWicketDelivery',y='bowler',data=x[:5],orient='h',width=0.4,ax=ax[0],color='#81C784')
    x_p.bar_label(x_p.containers[0])
    ax[0].set_ylabel('')
    ax[0].set_xlabel('')
    ax[0].set_title('Top Powerplay Strikers',fontsize=(15))
    
    y_p=sns.barplot(x='Avg',y='bowler',data=y[:5],orient='h',width=0.4,ax=ax[1],color='#EDE275')
    y_p.bar_label(y_p.containers[0])
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    ax[1].set_title('Best Powerplay Average',fontsize=(15))
    
    z_p=sns.barplot(x='Economy',y='bowler',data=z[:5],orient='h',width=0.4,ax=ax[2],color='#786D5F')
    z_p.bar_label(z_p.containers[0])
    ax[2].set_ylabel('')
    ax[2].set_xlabel('')
    ax[2].set_title('Best Powerplay Economy',fontsize=(15))
    
    m_p=sns.barplot(x='S/R',y='bowler',data=m[:5],orient='h',width=0.4,ax=ax[3],color='#E66C2C')
    m_p.bar_label(m_p.containers[0])
    ax[3].set_ylabel('')
    ax[3].set_xlabel('')
    ax[3].set_title('Best Powerplay Strike Rate',fontsize=(15))
    


# In[192]:


bow_dth_st()


# In[197]:


bat_vs_ball=deliveries.groupby(['batter','bowler'])['batsman_run'].sum().reset_index()


# In[200]:


bb_balls=deliveries.groupby(['batter','bowler'])['ballnumber'].count().reset_index()


# In[202]:


deliveries.groupby(['batter','bowler'])['isWicketDelivery'].count().reset_index()


# In[211]:


bat_vs_ball=bat_vs_ball.merge(bb_balls,left_on=['batter','bowler'],right_on=['batter','bowler'],how='outer')
bat_vs_ball=bat_vs_ball.merge(bb_wic,left_on=['batter','bowler'],right_on=['batter','bowler'],how='outer')


# In[213]:


bat_vs_ball.fillna(0,inplace=True)


# In[215]:


bat_vs_ball['isWicketDelivery']=bat_vs_ball['isWicketDelivery'].astype(int)


# In[216]:


bat_vs_ball['Batsman S/R']=np.round((bat_vs_ball['batsman_run']/bat_vs_ball['ballnumber'])*100,2)
bat_vs_ball['Bowler Avg']=np.round(bat_vs_ball['batsman_run']/bat_vs_ball['isWicketDelivery'],2)
bat_vs_ball['Bowler Economy']=np.round(bat_vs_ball['batsman_run']/(bat_vs_ball['ballnumber']//6),2)
bat_vs_ball['Bowler S/R']=np.round(bat_vs_ball['ballnumber']/bat_vs_ball['isWicketDelivery'],2)


# In[218]:


bat_vs_ball.replace(np.inf,0,inplace=True)


# In[220]:


bat_vs_ball.fillna(0,inplace=True)


# In[223]:


def batsman_vs_bowler(batsman,bowler):
    dt=bat_vs_ball[bat_vs_ball['batter']==batsman][bat_vs_ball['bowler']==bowler]
    return dt


# In[226]:


batsman_vs_bowler('RG Sharma','DJ Bravo')


# In[ ]:





# In[ ]:




