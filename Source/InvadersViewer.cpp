#include "LocalTraining.h"
#include "CnnAgent.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef EVOLUTION_HAS_SDL2
#include <SDL.h>

namespace {
struct SDLSession
{
    SDLSession() { if (SDL_Init(SDL_INIT_VIDEO) != 0) throw std::runtime_error(SDL_GetError()); }
    ~SDLSession() { SDL_Quit(); }
};
using Window = std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)>;
using Renderer = std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)>;

void rectangle(SDL_Renderer* renderer, int x, int y, int w, int h, int r, int g, int b)
{
    SDL_SetRenderDrawColor(renderer, r, g, b, 255);
    SDL_Rect rect{x, y, w, h};
    SDL_RenderFillRect(renderer, &rect);
}

void drawScore(SDL_Renderer* renderer, int score)
{
    // Tiny bitmap font keeps the HUD self-contained: no font files or SDL_ttf.
    static const byte_t digits[10][5] = {
        {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7},
        {5, 5, 7, 1, 1}, {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 1, 1, 1},
        {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7}
    };
    static const byte_t c[5] = {7, 4, 4, 4, 7};
    static const byte_t r[5] = {6, 5, 6, 5, 5};
    static const byte_t e[5] = {7, 4, 6, 4, 7};
    int x = 8;
    for (char character : "SCORE " + std::to_string(score))
    {
        const byte_t* glyph = nullptr;
        if (character >= '0' && character <= '9') glyph = digits[character - '0'];
        else if (character == 'S') glyph = digits[5];
        else if (character == 'C') glyph = c;
        else if (character == 'O') glyph = digits[0];
        else if (character == 'R') glyph = r;
        else if (character == 'E') glyph = e;
        if (glyph)
            for (int row = 0; row < 5; ++row)
                for (int col = 0; col < 3; ++col)
                    if (glyph[row] & (1 << (2 - col)))
                        rectangle(renderer, x + col * 2, 2 + row * 2, 2, 2, 220, 235, 255);
        x += 8;
    }
}

void draw(SDL_Renderer* renderer, const SpaceInvaders::World& world)
{
    SDL_SetRenderDrawColor(renderer, 8, 12, 22, 255);
    SDL_RenderClear(renderer);
    rectangle(renderer, 0, 235, 256, 1, 40, 100, 100);
    // Draw simple pixel sprites directly; the simulation never creates a framebuffer.
    for (int row = 0; row < SpaceInvaders::Rows; ++row)
        for (int col = 0; col < SpaceInvaders::Columns; ++col)
        {
            if (!(world.invaders[row] & (1u << col))) continue;
            const int x = SpaceInvaders::enemyX(world, col), y = SpaceInvaders::enemyY(world, row);
            const int green = 170 + row * 16;
            rectangle(renderer, x - 5, y - 3, 10, 6, 130, green, 255 - row * 30);
            rectangle(renderer, x - 3, y - 5, 6, 2, 130, green, 255 - row * 30);
            const int leg = (world.frame / 15) % 2;
            rectangle(renderer, x - 6 + leg, y + 3, 2, 3, 130, green, 255 - row * 30);
            rectangle(renderer, x + 4 - leg, y + 3, 2, 3, 130, green, 255 - row * 30);
            rectangle(renderer, x - 3, y - 1, 2, 2, 8, 12, 22);
            rectangle(renderer, x + 1, y - 1, 2, 2, 8, 12, 22);
        }
    for (int i = 0; i < SpaceInvaders::ShieldCells; ++i)
        if (world.shields[i])
            rectangle(renderer, SpaceInvaders::shieldX(i), SpaceInvaders::shieldY(i), 4, 4,
                      30, world.shields[i] == 2 ? 185 : 100, 90);
    if (world.lives && (!world.invulnerable || (world.frame / 5) % 2))
    {
        rectangle(renderer, world.playerX - 7, SpaceInvaders::PlayerY - 2, 14, 6, 70, 255, 170);
        rectangle(renderer, world.playerX - 2, SpaceInvaders::PlayerY - 6, 4, 4, 70, 255, 170);
    }
    if (world.playerShot.active)
        rectangle(renderer, world.playerShot.x - 1, world.playerShot.y - 3, 2, 6, 255, 255, 180);
    for (const auto& bullet : world.bullets)
        if (bullet.active) rectangle(renderer, bullet.x - 1, bullet.y - 2, 2, 5, 255, 110, 100);
    if (world.saucerActive)
    {
        rectangle(renderer, world.saucerX - 9, 16, 18, 4, 255, 100, 180);
        rectangle(renderer, world.saucerX - 5, 13, 10, 3, 255, 100, 180);
    }
    rectangle(renderer, 0, 0, SpaceInvaders::Width, 13, 8, 12, 22);
    drawScore(renderer, world.score);
    for (int i = 0; i < world.lives; ++i)
        rectangle(renderer, SpaceInvaders::Width - 44 + i * 12, 5, 8, 4, 70, 255, 170);
    SDL_RenderPresent(renderer);
}
}
#endif

int viewInvaders(const RunOptions& options, bool human)
{
#ifndef EVOLUTION_HAS_SDL2
    (void)options;
    (void)human;
    throw std::runtime_error("Viewer unavailable. Install SDL2 development files and configure with -DEVOLUTION_BUILD_VIEWER=ON.");
#else
    Genotype genotype;
    CnnGenome cnnGenome;
    if (!options.loadPath.empty())
    {
        if (options.agent == AgentType::Cnn) cnnGenome.load(options.loadPath);
        else genotype.load(options.loadPath);
    }
    InvadersAgent agent(genotype);
    CnnAgent cnnAgent(cnnGenome, options.seed);
    SpaceInvaders game(options.seed, options.maxFrames);
    SDLSession session;
    Window window(SDL_CreateWindow("Space Invaders", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  768, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE), SDL_DestroyWindow);
    if (!window) throw std::runtime_error(SDL_GetError());
    Renderer renderer(SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED), SDL_DestroyRenderer);
    if (!renderer) renderer.reset(SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_SOFTWARE));
    if (!renderer) throw std::runtime_error(SDL_GetError());
    SDL_RenderSetLogicalSize(renderer.get(), SpaceInvaders::Width, SpaceInvaders::Height);
    bool running = true, paused = false;
    int episode = 0;
    const std::string pilot = human ? "Human" : options.loadPath.empty() ? "Demo pilot" : "Agent: " + options.loadPath;
    const auto frameTime = std::chrono::microseconds(1000000 / options.fps);
    auto deadline = std::chrono::steady_clock::now();
    std::cout << "Viewer: " << pilot << "; P pause, N step, R restart, Esc quit\n";
    while (running)
    {
        bool singleStep = false;
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN && !event.key.repeat)
            {
                if (event.key.keysym.sym == SDLK_ESCAPE) running = false;
                if (event.key.keysym.sym == SDLK_p) paused = !paused;
                if (event.key.keysym.sym == SDLK_n) { paused = true; singleStep = true; }
                if (event.key.keysym.sym == SDLK_r)
                {
                    game.restart();
                    cnnAgent.reset(options.seed + static_cast<std::uint32_t>(episode));
                }
            }
        }
        if (!running) break;
        if (!paused || singleStep)
        {
            int action;
            if (human)
            {
                const auto* keys = SDL_GetKeyboardState(nullptr);
                const bool left = keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A];
                const bool right = keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D];
                const bool fire = keys[SDL_SCANCODE_SPACE];
                action = left == right ? (fire ? SpaceInvaders::Fire : SpaceInvaders::Noop) :
                         left ? (fire ? SpaceInvaders::LeftFire : SpaceInvaders::Left) :
                                (fire ? SpaceInvaders::RightFire : SpaceInvaders::Right);
            }
            else if (options.loadPath.empty()) action = game.demoAction();
            else action = options.agent == AgentType::Cnn ? cnnAgent.action(game) : agent.action(game);
            game.do_action(action);
        }
        const auto& state = game.world();
        const std::string title = "Space Invaders | " + pilot + " | Score " + std::to_string(state.score) +
                                 " | Lives " + std::to_string(state.lives) + " | Wave " + std::to_string(state.wave) +
                                 " | Frame " + std::to_string(state.frame) + (paused ? " | PAUSED" : "");
        SDL_SetWindowTitle(window.get(), title.c_str());
        draw(renderer.get(), state);
        if (game.is_over())
        {
            std::cout << "Episode " << episode + 1 << ": score=" << state.score << " frames=" << state.frame << '\n';
            if (++episode >= options.episodes) break;
            game.reset(options.seed + static_cast<std::uint32_t>(episode));
            cnnAgent.reset(options.seed + static_cast<std::uint32_t>(episode));
        }
        deadline += frameTime;
        const auto now = std::chrono::steady_clock::now();
        if (deadline < now) deadline = now;
        std::this_thread::sleep_until(deadline);
    }
    return 0;
#endif
}
