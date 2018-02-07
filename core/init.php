<?php
//Start Session
session_start();

// Include Configuration
require_once('config/config.php');

// Helper Function Files
require_once('helpers/system_helper.php');
require_once('helpers/format_helper.php');
require_once('helpers/db_helper.php');

/*
 * Create any classes you want in libraries,
 * as long as the file name matches class name,
 * it will be automatically required.
 */
// Autoload Classes
function __autoload($class_name) {
	require_once('libraries/'.$class_name.'.php');
}
