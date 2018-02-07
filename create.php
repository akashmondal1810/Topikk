<?php require 'core/init.php'; ?>
<?php 

// Create topic object
$topic = new Topic();

// Check if form was submitted
if (isset($_POST['do_create'])) {
	// Create Validator Object
	$validate = new Validator();
	
	// Create Data Array
	$data = array();
	$data['title'] = $_POST['title'];
	$data['body'] = $_POST['body'];
	$data['category'] = $_POST['category'];
	$data['user_id'] = getUser()['user_id'];
	$data['last_activity'] = date("Y-m-d H:j:s");
	
	// Required Fields
	$field_array = array('title', 'body', 'category');
	
	if ($validate->isRequired($field_array)) {
		if ($topic->create($data)) {
			redirect('index.php', 'Your topic has been posted', 'success');
		} else {
			redirect('topic.php?id='.$topic_id, 'Something went wrong with your post', 'error');
		}
	} else {
		redirect('create.php', 'Please fill in all required fields', 'error');
	}
}

// Get template
$template = new Template('templates/create.php');

//Assign vars
$template->title = 'Create New Topic';
$template->totalTopics = $topic->getTotalTopics();
// Display template
echo $template;
?>
