use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

type Job = Box<dyn FnOnce() + Send + 'static>;

enum Message {
    NewJob(Job),
    Terminate,
}

struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Message>,
}

impl ThreadPool {
    pub fn new(worker_count: usize) -> ThreadPool {
        let (sender, reciever) = mpsc::channel();
        let reciever = Arc::new(Mutex::new(reciever));

        let mut workers = Vec::<Worker>::with_capacity(worker_count);
        for i in 0..worker_count {
            let reciever = Arc::clone(&reciever);

            let worker = Worker {
                id: i,
                thread: Some(thread::spawn(move || loop {
                    let message = reciever.lock().unwrap().recv().unwrap();

                    match message {
                        Message::Terminate => break,
                        Message::NewJob(job) => job(),
                    }
                })),
            };
            workers.push(worker);
        }

        return ThreadPool { workers, sender };
    }

    pub fn go<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);

        self.sender.send(Message::NewJob(job)).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in self.workers.iter() {
            self.sender.send(Message::Terminate).unwrap();
        }

        for worker in self.workers.iter_mut() {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}
