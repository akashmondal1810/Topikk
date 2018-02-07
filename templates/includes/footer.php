					</div>
				</div>
			</div>
			
			<div class="col-md-3">
				<div class="sidebar">
					<div class="block">
						<?php if (isLoggedIn()) : ?>
						<div class="userdata">
							<img class="avatar" src="<?php echo BASE_URI.'/images/avatars/'.$_SESSION['avatar'] ?>">
							<div align="center"><h3><?php echo getUser()['username'] ?></h3></div>
						</div>
						<br>
						<div align="center">
						<form role="form" method="post" action="logout.php" class="display-inline">
							<input type="submit" name="do_logout" class="btn btn-primary" value="Log Out" />
						</form>
						<form role="form" method="post" action="editprofile.php">
							<input type="submit" name="do_profile" class="btn" value="Edit Profile" />
						</form>
						</div>
						<?php else : ?>
						<h3>Login Form</h3>
						<form role="form" method="post" action="login.php">
						<div class="form-group">
							<label>Username</label>
							<input name="username" type="text" class="form-control" placeholder="Enter Username">
						</div>
						<div class="form-group">
							<label>Password</label>
							<input name="password" type="password" class="form-control" placeholder="Enter Password">
						</div>			
						<button name="do_login" type="submit" class="btn btn-primary">Login</button>
						<a  class="btn btn-default" href="register.php"> Create Account</a>
					</form>
					<?php endif; ?>
					</div>
					
					<div class="block">
					<h3>Categories</h3>
					<div class="list-group">
						<a href="topics.php" class="list-group-item <?php echo is_active(null) ?>">All Topics <span class="badge pull-right"><?php echo $totalTopics ?></span></a>
						<?php foreach (getCategories() as $category) : ?>
						<a href="topics.php?category=<?php echo $category->id ?>" class="list-group-item <?php echo is_active($category->id) ?>"><?php echo $category->name ?><span class="badge pull-right"><?php echo categoryTopicCount($category->id) ?></span></a>
						<?php endforeach; ?>
					</div>
				</div>	
				</div>
			</div>
		</div>
    </div><!-- /.container -->
  </body>
</html>
