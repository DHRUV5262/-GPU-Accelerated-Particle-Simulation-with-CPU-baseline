/** Thin launcher so the exe has exactly one source with main() (fixes VS+CUDA link). */
extern int run_visualizer(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    return run_visualizer(argc, argv);
}
