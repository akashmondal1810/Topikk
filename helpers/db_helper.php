<?php
/*
 * Get # of replies per topic
 */
function replyCount($topic_id){
	$db = new Database();
	$db->query('SELECT * FROM replies WHERE topic_id = :topic_id');
	$db->bind(':topic_id', $topic_id);
	// Assign Rows
	$db->resultset();
	// Get Count
	return $db->rowCount();
}

/*
 * Get # of topics by category
 */
function categoryTopicCount($category_id){
	$db = new Database();
	$db->query('SELECT * FROM `topics` WHERE topics.category_id = :category_id');
	$db->bind(':category_id', $category_id);
	$db->resultset();
	return $db->rowCount();
}

/*
 * Get Categories
 */
function getCategories(){
	$db = new Database();
	$db->query('SELECT * FROM `categories`');
	$results = $db->resultset();
	return $results;
}

/*
 * Get user post count
 */
function userPostCount($user_id){
	// Get post count from topics
	$db = new Database();
	$db->query("SELECT * FROM topics WHERE user_id = :user_id");
	$db->bind(':user_id', $user_id);
	$rows = $db->resultset();
	$topic_count = $db->rowCount();
	
	// Get post count from replies
	$db = new Database();
	$db->query("SELECT * FROM replies WHERE user_id = :user_id");
	$db->bind(':user_id', $user_id);
	$rows = $db->resultset();
	$reply_count = $db->rowCount();
	
	return $topic_count + $reply_count;
}
