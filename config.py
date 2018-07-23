# coding: utf-8

CSV_PATH = {
    'twitter-facebook' : { 
        'tnet' : '../tt-fb/twitter_network.csv',
        'fnet' : '../tt-fb/facebook_network.csv',
        'tuname' : '../tt-fb/twitter_username.csv',
        'funame' : '../tt-fb/facebook_username.csv',
        'align' : '../tt-fb/alignment.csv'
    },
    'douban-weibo' : {}
}

JSON_PATH = {
    'twitter-facebook' : { 
        'tnet' : '../tt-fb/twitter_network.json',
        'fnet' : '../tt-fb/facebook_network.json',
        'tuname' : '../tt-fb/twitter_username.json',
        'funame' : '../tt-fb/facebook_username.json',
        'align' : '../tt-fb/facebook2twitter.json'
    },
    'douban-weibo' : {}
}

JSON_NEW_PATH = {
    'twitter-facebook' : { 
        'tnet' : '../tt-fb/twitter_network_new.json',
        'tnet_raw2new' : '../tt-fb/twitter_network_raw2new.json',
        'tnet_new2raw' : '../tt-fb/twitter_network_new2raw.json',
        'fnet' : '../tt-fb/facebook_network_new.json',
        'fnet_raw2new' : '../tt-fb/facebook_network_raw2new.json', 
        'fnet_new2raw' : '../tt-fb/facebook_network_new2raw.json',
        'tuname' : '../tt-fb/twitter_username_new.json',
        'funame' : '../tt-fb/facebook_username_new.json',
        'align' : '../tt-fb/facebook2twitter_new.json'
    },
    'douban-weibo' : {}
}

RANDOM_WALK_CSV_PATH = {
    'twitter-facebook' : { 
        'tnet' : '../tt-fb/twitter_network_random_walk.csv',
        'fnet' : '../tt-fb/facebook_network_random_walk.csv',
    },
    'douban-weibo' : {}
}

LOG_PATH = { 
    'twitter-facebook' : { 
        'tnet' : '../log/twitter_network.log',
        'fnet' : '../log/facebook_network.log',
        'tuname' : '../log/twitter_username.log',
        'funame' : '../log/facebook_username.log',
        'align' : '../log/facebook2twitter.log'
    },
    'douban-weibo' : {}
}

USERS_NUMS = {
    'fb' : 422290+10,
    'tt' : 669197+10,
}


