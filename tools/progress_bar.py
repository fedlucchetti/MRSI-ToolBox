from rich.progress import Progress
import threading
import time, sys

class ProgressBar:
    def __progress_bar_task(self, duration):
        with Progress() as progress:
            # Create a task with a total of `duration` seconds
            task = progress.add_task("[green]Registration...", total=duration)
            
            # Update the progress bar every second
            for _ in range(duration):
                time.sleep(1)  # Wait for a second
                progress.update(task, advance=1)  # Advance the progress bar by 1

    def run_progress_bar_in_background(self, duration):
        # Create a thread that runs the progress bar task
        time.sleep(1)
        thread = threading.Thread(target=self.__progress_bar_task, args=(duration,), daemon=True)
        thread.start()
        # For debugging, wait for the thread to complete
        # Remove or comment out the line below in actual use to run the thread in the background
        return thread

# Example usage
if __name__ == "__main__":
    # if sys.argv[1]==None:
    #     duration = 10
    # else:
    duration = int(sys.argv[1])
    demo = ProgressBar()
    demo.run_progress_bar_in_background(duration)
    # If you remove thread.join(), ensure the main thread stays alive long enough
    # e.g., by adding a sleep call or waiting for user input
    time.sleep(duration+1)  # Ensure main program runs long enough to see progress
