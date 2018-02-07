<?php
/*
 * Recieves form info, validates, redirects
 */

include 'core/init.php';

if (isset($_POST['do_login'])) {
	// Get vars
	$username = $_POST['username'];
	$password = md5($_POST['password']);
	
	// Create User Object
	$user = new User();
	
	if ($user->login($username, $password)) {
		redirect('index.php', "You are now logged in as <strong>$username</strong>", 'success');
	} else {
		// If login fails
		redirect('index.php', 'That login is not valid', 'error');
	}
} else {
	redirect('index.php');
}
