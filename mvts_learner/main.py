"""
Main entrypoint for the MVTS-Learner application, provide a way to run the app locally, as a server, or as a client
using the command line.
"""
import argparse
import logging
import os

from mvts_learner.options.options import framework_option_class_deducer
from mvts_learner.run_target import framework_target_function

log = logging.getLogger(__name__)
# pylint: disable=import-outside-toplevel

def main(*args, debug_level=logging.INFO):
	"""
	The main-function. Used as an entrypoint for the application.
	"""
	argparser = argparse.ArgumentParser()
	argparser.add_argument("mode",
		choices=["local", "server", "client"],
		help="Whether to run the app enitrely locally, as a server, or as a client"
	)
	argparser.add_argument(
		"--port",
		type=int,
		default=5353,
		help="Port to run the server on. Only used if mode is server"
	)
	argparser.add_argument(
		"--workspace_path",
		type=str,
		default=os.path.join(os.getcwd(), "Configurun-Workspaces"),
		help="Path to the workspace directory. If it doesn't exist, it will be created. Defaults to ./Configurun-Workspaces"
	)
	argparser.add_argument(
		"--pw_path",
		type=str,
		default=None
	)
	argparser.add_argument(
		"--debug",
		action="store_true",
		help="Whether to run in debug mode"
	)
	args = argparser.parse_args()

	if args.debug:
		debug_level = logging.DEBUG

	workspace_path = args.workspace_path
	if workspace_path == os.path.join(os.getcwd(), "Configurun-Workspaces"): #If default, also append the mode
		os.makedirs(workspace_path, exist_ok=True)
		workspace_path = os.path.join(args.workspace_path, f"{args.mode}-workspace")

	workspace_path = os.path.abspath(workspace_path)

	#Remove last folder from workspace path and check if it exists
	workspace_parent_path = os.path.dirname(workspace_path)
	if not os.path.exists(workspace_parent_path):
		raise FileNotFoundError(f"Parent directory of workspace path (--workspace_path=){workspace_path} must exist.")

	os.makedirs(workspace_path, exist_ok=True)

	pw_path = args.pw_path
	password = None

	if args.mode == "server" and args.pw_path is None:
		#Create pw-file if it doesn't exist
		pw_path = os.path.join(workspace_path, "pw.txt")
		if not os.path.exists(pw_path):
			log.warning(f"Passed password file (--pw_path=){pw_path} does not exist. "
	       		"Please insert a new password (note that it is stored in the workspace-path)? [y/n]")

			set_pw1, set_pw2 = "a", "b"
			while set_pw1 != set_pw2:
				set_pw1 = input("Enter password: ")
				set_pw2 = input("Confirm password: ")
				if set_pw1 != set_pw2:
					log.warning("Passwords do not match. Please try again.")

			with open(pw_path, "w", encoding="utf-8") as file:
				log.info(f"Writing password to {pw_path}")
				file.write(set_pw1)

		with open(pw_path, "r", encoding="utf-8") as file:
			password = file.read()

		assert password is not None, "Password is None, but should be set"
		assert len(password) > 0, f"Password-file {pw_path} is empty, but should be set"

	log.info(f"Workspace path: {workspace_path}")

	if args.mode == "local":
		# Run the Configurun app locally
		from configurun.app import run_local
		log.info("Now starting local run")
		# import tempfile
		# tempdir = tempfile.gettempdir()
		run_local(
			target_function=framework_target_function,
			options_source=framework_option_class_deducer,
			workspace_path=workspace_path,
			log_level=debug_level
		)
		log.info("Now done with local run")


	elif args.mode == "server":
		# Run the configurun app as a server
		assert password is not None, "Password is None, but should be set when running server..."
		log.info("Now starting server")
		from configurun.server import run_server
		run_server(
			target_function=framework_target_function,
			workspace_path=workspace_path,
			log_level=debug_level,
			password=password,
			port = args.port
		)
		log.info("Now done with server")

	elif args.mode == "client":
		# Run the configurun app as a client
		from configurun.app import run_client
		log.info("Now starting client")
		run_client(
			options_source=framework_option_class_deducer,
			workspace_path=workspace_path,
			log_level=debug_level
		)
		log.info("Now done with client")

	log.info("Done with main.")



if __name__ == "__main__":
	main()
