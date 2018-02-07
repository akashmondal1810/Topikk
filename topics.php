<?php require 'core/init.php'; ?>
<?php 
// Create topics object
$topic = new Topic;

// Create User Object
$user = new User();

// Get category from URL
$category = isset($_GET['category']) ? $_GET['category'] : null;

// Get user from URL
$user_id = isset($_GET['user']) ? $_GET['user'] : null;

// Get template and assign vars
$template = new Template('templates/topics.php');

//Assign vars
$template->totalTopics = $topic->getTotalTopics();
$template->totalCategories = $topic->getTotalCategories();
$template->totalUsers = $user->getTotalUsers();

// Check for category filter
if(isset($category)){
	$template->topics = $topic->getByCategory($category);
	$template->title = 'Posts In "'.$topic->getByCategory($category)['0']->name.'"';
}
	
// Check for user filter
if(isset($user_id)){
	$template->topics = $topic->getByUser($user_id);
	$template->title = 'Posts By '.$topic->getNameById($user_id)->username;
}

if(!isset($category) && !isset($user_id)){
	$template->topics = $topic->getAllTopics();
}

// Display template
echo $template;
?>
