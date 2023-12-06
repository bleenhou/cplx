package eth.bram.cplx;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class Utils {

	/**
	 * Gets a future, wrapping exceptions in {@link RuntimeException}. 
	 */
	public static <V> V ensureFinished(Future<V> f){
		try {
			return f.get();
		} catch (InterruptedException | ExecutionException e) {
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * Safely put in a queue
	 */
	public static <V> void safePut(ArrayBlockingQueue<V> queue, V item) {
		try {
			queue.put(item);
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * Safely take from a queue
	 */
	public static <V> V safeTake(ArrayBlockingQueue<V> queue) {
		try {
			return queue.take();
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
	}
}
